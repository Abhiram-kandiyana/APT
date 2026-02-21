"""
APT–MDL: Minimum-Description-Length Active Prompt Tuning
Dr. MAli overleaf
"""
import numpy as np
import random
from typing import List, Tuple, Any, Dict, Union, Optional
from collections import Counter
import base64
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import argparse
import json
import subprocess
import tempfile
import sys
import shutil
import hashlib
from numbers import Integral
from pathlib import Path
from datetime import datetime
from utils import *

# Initialize Tokenizer (globally to avoid reloading)
# Using cl100k_base which is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
enc = tiktoken.get_encoding("cl100k_base")

# Lazily initialized so DTS mode does not touch HuggingFace tokenizers at import time.
st_model = None

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Short alias -> full Hugging Face model id for DTS image embeddings.
DTS_CLIP_MODEL_ALIASES: Dict[str, str] = {
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "clip": "openai/clip-vit-base-patch32",
    "phikonv2": "owkin/phikon-v2",
    "medsiglip": "google/medsiglip-448",
}
DEFAULT_DTS_CLIP_MODEL_ALIAS = "biomedclip"
DEFAULT_DTS_CLIP_MODEL_NAME = DTS_CLIP_MODEL_ALIASES["biomedclip"]


# ============================================================
# TYPE DEFINITIONS
# ============================================================

Image = Any          # e.g., numpy array, PIL image, or path string
Caption = str
Label = int

PromptExemplar = Tuple[Image, Caption]     # (x, c)
PromptSet = List[PromptExemplar]
LabeledExample = Tuple[Image, Label]


# ============================================================
# USER-PROVIDED COMPONENTS (Dr. Mali specific components)
# ============================================================

def load_files(file_path: str) -> str:
    """
    Load content from a text or markdown file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def with_selection_suffix(file_path: str, selection_method: str) -> str:
    """
    Add the selection method to artifact filenames to avoid method collisions.
    Example: logs/vlm_logs.json -> logs/vlm_logs_mdl.json
    """
    if not file_path:
        return file_path

    selection = str(selection_method).lower().strip()
    suffix_token = f"_{selection}"
    path_obj = Path(file_path)

    if path_obj.stem.endswith(suffix_token):
        return str(path_obj)

    return str(path_obj.with_name(f"{path_obj.stem}{suffix_token}{path_obj.suffix}"))


def _safe_path_token(value: Any, default: str = "unknown") -> str:
    token = str(value).strip() if value is not None else ""
    if not token:
        token = default
    token = re.sub(r"[^A-Za-z0-9._=-]+", "_", token)
    token = token.strip("._-")
    return token or default


_UNCERTAINTY_CACHE_MEM: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _resolve_uncertainty_cache_path(**gen_kwargs) -> str:
    cache_path = str(gen_kwargs.get("uncertainty_cache_path", "logs/uncertainty_cache.jsonl") or "").strip()
    if not cache_path:
        cache_path = "logs/uncertainty_cache.jsonl"
    return cache_path


def _load_uncertainty_cache_entries(cache_path: str) -> Dict[str, Dict[str, Any]]:
    if cache_path in _UNCERTAINTY_CACHE_MEM:
        return _UNCERTAINTY_CACHE_MEM[cache_path]

    entries: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    key = str(payload.get("key", "")).strip()
                    if not key:
                        continue
                    entries[key] = payload
        except Exception as e:
            print(f"Warning: failed to load uncertainty cache {cache_path}: {e}")
            entries = {}

    _UNCERTAINTY_CACHE_MEM[cache_path] = entries
    return entries


def _append_uncertainty_cache_entry(cache_path: str, entry: Dict[str, Any]) -> None:
    entries = _load_uncertainty_cache_entries(cache_path)
    key = str(entry.get("key", "")).strip()
    if not key:
        return

    entries[key] = dict(entry)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry))
        f.write("\n")


def _prompt_set_signature(prompt_set: PromptSet) -> str:
    serialized = [(str(x), str(c)) for x, c in prompt_set]
    blob = json.dumps(serialized, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _uncertainty_cache_key(
    image_path: str,
    prompt_set: PromptSet,
    K: int,
    **gen_kwargs,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    round_num = int(gen_kwargs.get("round_num", 0) or 0)
    dataset = str(gen_kwargs.get("dataset", "") or "")
    fold = gen_kwargs.get("fold", None)
    model = str(gen_kwargs.get("model", "gpt-4o"))
    temperature = float(gen_kwargs.get("temperature", 1))
    top_p = float(gen_kwargs.get("top_p", 1.0))
    max_tokens = int(gen_kwargs.get("max_tokens", 1000))
    timeout_s = float(gen_kwargs.get("vlm_timeout_s", 120.0))

    prompt_signature = None
    prompt_mode = "round1_no_prompt_check"
    if round_num > 1:
        prompt_signature = _prompt_set_signature(prompt_set)
        prompt_mode = "prompt_signature_required"

    key_payload = {
        "dataset": dataset,
        "fold": fold,
        "round": int(round_num),
        "image_path": os.path.abspath(str(image_path)),
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "vlm_timeout_s": timeout_s,
        "K_uncertainty": int(K),
        "prompt_mode": prompt_mode,
        "prompt_signature": prompt_signature if prompt_signature is not None else "",
    }
    key_blob = json.dumps(key_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    key = hashlib.sha256(key_blob.encode("utf-8")).hexdigest()
    return key, prompt_signature, key_payload


def _selection_artifact_token(
    selection_method: str,
    dts_clip_model_name: Optional[str] = None,
    active_set_batch_size: Optional[int] = None,
    candidate_pool_size: Optional[int] = None,
) -> str:
    selection = str(selection_method or "mdl").lower().strip()
    if selection != "dts":
        return _safe_path_token(selection, default="mdl")

    batch_token = ""
    try:
        b = int(active_set_batch_size) if active_set_batch_size is not None else None
        if b is not None and b > 0:
            batch_token = f"_b={b}"
    except (TypeError, ValueError):
        batch_token = ""
    cand_token = ""
    try:
        c = int(candidate_pool_size) if candidate_pool_size is not None else -1
        cand_token = f"_candidate-size={c}"
    except (TypeError, ValueError):
        cand_token = ""

    model_alias = _dts_clip_model_alias(dts_clip_model_name)
    if model_alias == "biomedclip":
        return f"dts_biomedclip{batch_token}{cand_token}"
    if model_alias == "clip":
        return f"dts_clip{batch_token}{cand_token}"
    if model_alias == "phikonv2":
        return f"dts_phikonv2{batch_token}{cand_token}"
    if model_alias == "medsiglip":
        return f"dts_medsiglip{batch_token}{cand_token}"

    model_name = str(dts_clip_model_name or "").strip().lower()
    return f"dts_{_safe_path_token(model_name or 'clip', default='clip')}{batch_token}{cand_token}"


def _dts_clip_model_alias(dts_clip_model_name: Optional[str]) -> Optional[str]:
    model_name = str(dts_clip_model_name or "").strip()
    if not model_name:
        return None
    key = model_name.lower()
    if key in DTS_CLIP_MODEL_ALIASES:
        return key
    for alias, full_name in DTS_CLIP_MODEL_ALIASES.items():
        if key == str(full_name).lower():
            return alias
    return None


def _resolve_dts_clip_model_name(dts_clip_model_name: Optional[str]) -> str:
    model_name = str(dts_clip_model_name or "").strip()
    if not model_name:
        return DEFAULT_DTS_CLIP_MODEL_NAME
    alias = _dts_clip_model_alias(model_name)
    if alias is not None:
        return DTS_CLIP_MODEL_ALIASES[alias]
    return model_name


def _resolve_dts_clip_model_from_inputs(
    dts_clip_model_alias: Optional[str],
    dts_clip_model_name: Optional[str],
) -> str:
    alias = str(dts_clip_model_alias or "").strip().lower()
    if alias:
        if alias not in DTS_CLIP_MODEL_ALIASES:
            valid = ", ".join(sorted(DTS_CLIP_MODEL_ALIASES.keys()))
            raise ValueError(f"Invalid dts_clip_model_alias='{alias}'. Valid aliases: {valid}")
        return DTS_CLIP_MODEL_ALIASES[alias]
    return _resolve_dts_clip_model_name(dts_clip_model_name)


def build_dts_run_output_dir(
    base_outdir: str,
    dataset_name: str,
    fold: Optional[int],
    selection_token: str,
) -> str:
    """
    Build a unique per-run diagnostics directory:
    diagnostics/<dataset>_fold-<fold>_<timestamp>[/_<n>]
    """
    dataset_token = _safe_path_token(dataset_name, default="dataset")
    fold_token = _safe_path_token(fold, default="na")
    selection_token = _safe_path_token(selection_token, default="dts")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{dataset_token}_fold-{fold_token}_{selection_token}_{timestamp}"
    root = str(base_outdir or "diagnostics")

    candidate = os.path.join(root, prefix)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(root, f"{prefix}_{suffix}")
        suffix += 1
    return candidate


def resolve_fold_random_seed(dataset_name: str, fold: Optional[int], datasets_root: str = "datasets") -> Optional[int]:
    """
    Read fold-specific random seed from datasets/<dataset>/splits.json.
    Expected shape:
      { "<fold>": { "random seed": <int>, ... }, ... }
    """
    if fold is None:
        return None

    candidate_roots = [
        Path(datasets_root),
        Path(__file__).resolve().parent / "datasets",
        Path(__file__).resolve().parent.parent / "APT-MDL" / "datasets",
    ]
    splits_path = None
    for root in candidate_roots:
        candidate = root / str(dataset_name) / "splits.json"
        if candidate.exists():
            splits_path = candidate
            break
    if splits_path is None:
        return None

    try:
        with open(splits_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    fold_key = str(fold)
    fold_entry = data.get(fold_key) if isinstance(data, dict) else None
    if not isinstance(fold_entry, dict):
        return None

    for key in ("random seed", "random_seed", "seed"):
        if key in fold_entry:
            try:
                return int(fold_entry[key])
            except Exception:
                return None
    return None


def set_global_random_seed(seed: int) -> None:
    """Set global RNG seeds for Python/NumPy/Torch (if available)."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ["PYTHONHASHSEED"] = str(int(seed))

    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # Prefer deterministic behavior when possible.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    except Exception:
        # Torch may not be installed for non-DTS runs.
        pass


def resolve_oracle_dataset_name(dataset_name: str, label_map: List[str]) -> str:
    """
    Map local dataset naming to oracle.py dataset naming conventions.
    """
    ds = (dataset_name or "").strip()
    ds_lower = ds.lower()
    labels_lower = {str(lbl).lower() for lbl in (label_map or [])}

    if "lurcher" in ds_lower or labels_lower == {"wild", "lurcher"}:
        return "Lurcher"
    if "fengru" in ds_lower:
        return "FengRu"
    return ds or "Lurcher"


def build_round_prompt_paths(
    prompts_root: str,
    dataset_name: str,
    fold: Union[int, None],
    round_num: Union[int, str],
    selection_method: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Build per-round prompt artifact paths:
    prompt_sets/<dataset>/fold-<fold>_<selection>/fold<fold>_round<round>_prompts_<selection>.json
    prompt_sets/<dataset>/fold-<fold>_<selection>/fold<fold>_round<round>_prompts_<selection>_corrected.json
    """
    round_token = str(round_num)
    dataset_token = _safe_path_token(dataset_name, default="dataset")
    fold_token = _safe_path_token(fold, default="na")
    selection_token = _safe_path_token(
        (selection_method or "mdl").lower().strip(),
        default="mdl",
    )
    fold_dir = os.path.join(prompts_root, dataset_token, f"fold-{fold_token}_{selection_token}")
    os.makedirs(fold_dir, exist_ok=True)

    if fold is not None:
        stem = f"fold{fold}_round{round_token}_prompts"
    else:
        stem = f"round{round_token}_prompts"

    stem = f"{stem}_{selection_token}"
    input_path = os.path.join(fold_dir, f"{stem}.json")
    output_path = os.path.join(fold_dir, f"{stem}_corrected.json")
    return input_path, output_path


def to_class_rooted_relative_path(image_path: str, label_map: List[str]) -> str:
    """
    Convert an image path into a class-rooted relative path.
    The first directory segment must be an exact class-name match (case-insensitive),
    e.g., "wild/..." or "lurcher/...".
    """
    raw = str(image_path or "").strip()
    if not raw:
        return raw

    # Normalize separators first so mixed path styles are handled consistently.
    parts = [p for p in re.split(r"[\\/]+", raw) if p and p != "."]
    class_names = {str(lbl).strip().lower() for lbl in (label_map or [])}

    for idx, segment in enumerate(parts):
        if segment.lower() in class_names:
            return "/".join(parts[idx:])

    # Fallback: keep path relative (no leading slash), but warn that class-rooting failed.
    fallback = "/".join(parts)
    print(
        f"Warning: Could not find class-rooted segment in path '{raw}'. "
        f"Storing fallback relative path '{fallback}'."
    )
    return fallback

def log_vlm_response(log_path: str, image_path: Union[str, List[str]], label: Union[Any, List[Any]], rationale: Union[str, List[str]] = None):
    """
    Appends the VLM response to a JSON file.
    Handles both single and batched inputs.
    Checks if image_path already exists; if so, updates label and rationale.
    """
    # Read existing data or initialize empty list
    data = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    data = json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {log_path}. Starting fresh.")
            data = []

    # Create a lookup map for existing entries: image_path -> index
    existing_map = {entry["image_path"]: i for i, entry in enumerate(data)}

    entries_to_process = []
    
    if isinstance(image_path, list):
        # Batch logging
        # Check if label is a list of tuples (parsed results)
        if isinstance(label, list) and len(label) > 0 and isinstance(label[0], tuple):
             for img, (lbl, rat) in zip(image_path, label):
                 entries_to_process.append({
                     "image_path": str(img),
                     "label": lbl,
                     "rationale": rat
                 })
        else:
            # Fallback if passed as separate lists
            # Handle case where rationale might be None
            current_rationale = rationale if isinstance(rationale, list) else [None] * len(image_path)
            
            for img, lbl, rat in zip(image_path, label, current_rationale):
                entries_to_process.append({
                    "image_path": str(img),
                    "label": lbl,
                    "rationale": rat
                })
    else:
        # Single entry
        entries_to_process.append({
            "image_path": str(image_path),
            "label": label,
            "rationale": rationale
        })
    
    # Process entries
    for entry in entries_to_process:
        img_path = entry["image_path"]
        if img_path in existing_map:
            # Update existing
            idx = existing_map[img_path]
            data[idx]["label"] = entry["label"]
            data[idx]["rationale"] = entry["rationale"]
        else:
            # Add new
            data.append(entry)
            # Update map in case of duplicates within the current batch
            existing_map[img_path] = len(data) - 1
    
    # Write back to file
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def vlm_query(
    x: Union[Image, List[Image]],
    system_prompt_1: str,
    system_prompt_2_template: str,
    prompt_set: PromptSet,
    stochastic: bool = False,
    **gen_kwargs
) -> Union[Tuple[Label, Caption], List[Tuple[Label, Caption]]]:
    """
    Wrapper around your Vision–Language Model (VLM).

    Parameters:
        x: image (path to image file) or list of images
        system_prompt_1: content of system prompt 1
        system_prompt_2_template: template string for system prompt 2
        prompt_set: list of exemplars [(x_i, c_i)]
        stochastic: if True, sampling with temperature for uncertainty
        gen_kwargs: any additional generation parameters

    Returns:
        (y_hat, e_hat): predicted label and rationale (or list of tuples)
    """
    # Determine if input is a batch
    is_batch = isinstance(x, list)
    images = x if is_batch else [x]
    
    # 1. Construct the message content
    content = []
    
    sp1_text = system_prompt_1
    # Format system prompt 2 with batch size
    sp2_text = system_prompt_2_template.format(N=len(images))

    # System prompt 1
    content.append({
        "type": "text",
        "text": sp1_text
    })

    # Few-shot exemplars
    for x_i, c_i in prompt_set:
        base64_img = encode_image(x_i)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
        })
        content.append({
            "type": "text",
            "text": c_i # consistent with APT
        })

    # System prompt 2 (after prompt_set)
    content.append({
        "type": "text",
        "text": sp2_text
    })

    # Query image(s)
    for img in images:
        base64_x = encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_x}"}
        })

    # 2. Call OpenAI API with retries
    # Default parameters
    model = gen_kwargs.get("model", "gpt-4o")
    temperature = gen_kwargs.get("temperature", 1)
    max_tokens = gen_kwargs.get("max_tokens", 1000)
    top_p = gen_kwargs.get("top_p", 1.0)
    request_timeout_s = float(gen_kwargs.get("vlm_timeout_s", 120))

    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=request_timeout_s,
            )
            output_text = response.choices[0].message.content
            lower_stripped = output_text.lower().strip()
            if lower_stripped == "" or ("i'm" in lower_stripped and ("can't" in lower_stripped or "can’t" in lower_stripped or "cannot" in lower_stripped or "unable" in lower_stripped)) or ("I'm unable" in lower_stripped):
                print("Model returned an invalid/refusal response; retrying after delay.")
                retry_count += 1
                time.sleep(2)
                continue
            
            # 3. Parse output
            parsed_results = parse_vlm_response(output_text, gen_kwargs)
            
            # Validation for batch processing
            if len(parsed_results) != len(images):
                print(f"Warning: Mismatch in number of predictions ({len(parsed_results)}) vs images ({len(images)}). Retrying...")
                retry_count += 1
                time.sleep(1)
                continue
            
            # Log response if path provided
            vlm_log_path = gen_kwargs.get("vlm_log_path")
            if vlm_log_path:
                try:
                    log_vlm_response(vlm_log_path, images, parsed_results)
                except Exception as log_e:
                    print(f"Warning: Failed to log VLM response to {vlm_log_path}: {log_e}")
                
            if is_batch:
                return parsed_results
            else:
                return parsed_results[0]
                
        except Exception as e:
            print(f"Error calling VLM (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            time.sleep(2)
            
    # If retries exhausted, return invalid prediction so metrics can ignore it.
    print(f"Max retries ({max_retries}) exhausted. Returning invalid label -1.")
    default_label = -1
    default_rationale = (
        "Unable to get a classification from the model even after multiple retries. "
        "This prediction is marked invalid and should be excluded from accuracy metrics."
    )
    default_results = [(default_label, default_rationale) for _ in images]
    
    # Log default response if path provided
    vlm_log_path = gen_kwargs.get("vlm_log_path")
    if vlm_log_path:
        try:
            log_vlm_response(vlm_log_path, images, default_results)
        except Exception as log_e:
            print(f"Warning: Failed to log default VLM response to {vlm_log_path}: {log_e}")

    if is_batch:
        return default_results
    else:
        return default_results[0]


def parse_vlm_response(output_text: str, gen_kwargs: Dict[str, Any]) -> List[Tuple[Label, Caption]]:
    """
    Parses the VLM output text to extract labels and rationales.
    Returns a list of (label, rationale) tuples.
    """
    results = []
    
    # Split by "R:" to handle multiple outputs
    # The format is expected to be R: ... C: ... repeated
    # We can use regex to find all occurrences
    
    # Pattern to match R: ... C: ... blocks
    # Using non-greedy match for rationale
    pattern = r"['\"]?R['\"]?:\s*(.*?)\s*['\"]?C['\"]?:\s*['\"]?(\w+)['\"]?"
    matches = re.findall(pattern, output_text, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        # Fallback for single output or malformed output
        # Try previous logic for single item
        rationale = "" 
        label = 0
        
        r_match = re.search(r"['\"]?R['\"]?:\s*(.*?)(?=['\"]?C['\"]?:|$)", output_text, re.DOTALL | re.IGNORECASE)
        if r_match:
            rationale = r_match.group(1).strip()
        else:
            rationale = output_text # Fallback

        c_match = re.search(r"['\"]?C['\"]?:\s*['\"]?(\w+)['\"]?", output_text, re.IGNORECASE)
        if c_match:
            label_str = c_match.group(1).upper()
            label = map_label(label_str, gen_kwargs)
        
        results.append((label, rationale))
    else:
        for r_text, c_text in matches:
            rationale = r_text.strip()
            label_str = c_text.upper()
            label = map_label(label_str, gen_kwargs)
            results.append((label, rationale))
            
    return results

def map_label(label_str: str, gen_kwargs: Dict[str, Any]) -> int:
    # Check for label mapping in gen_kwargs
    label_map = gen_kwargs.get("label_map", None)
    label = 0
    
    if label_map and isinstance(label_map, list):
        try:
            label_map_upper = [l.upper() for l in label_map]
            if label_str in label_map_upper:
                label = label_map_upper.index(label_str)
            else:
                label = 0 
        except ValueError:
            label = 0
    else:
        if label_str in ['T', 'P', '1']:
            label = 1
        elif label_str in ['U', 'W', '0']:
            label = 0
        else:
            try:
                label = int(label_str)
            except:
                label = 0 
    return label


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def text_encoder(captions: List[Caption]) -> np.ndarray:
    """
    Encode captions into embedding vectors.

    Returns:
        np.ndarray with shape [len(captions), d].

    NOTE: You MUST implement this using SentenceTransformers,
          Instructor, MiniLM, or any embedding model.
    """
    if not captions:
        return np.zeros((0, 384)) # 384 is dimension of all-MiniLM-L6-v2

    global st_model
    if st_model is None:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = st_model.encode(captions)
    return embeddings


def tokenizer(c: Caption) -> List[int]:
    """
    BPE tokenizer using tiktoken (cl100k_base).
    Returns a list of token IDs.
    """
    return enc.encode(c)


# ============================================================
# CAPTION COMPLEXITY & DL(P)
# ============================================================

def caption_complexity(
    c: Union[Caption, List[Caption]],
    prompt_set: PromptSet,
    prompt_embeddings: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.1,
) -> Union[float, List[float]]:
    """
    C(c;P) = alpha * len(c) + beta * max_sim(psi(c), psi(c')).
    Supports batch processing.
    """
    is_batch = isinstance(c, list)
    captions = c if is_batch else [c]

    # Batch tokenization
    lengths = [len(tokenizer(cap)) for cap in captions]
    length_penalties = [alpha * L for L in lengths]

    if len(prompt_set) == 0:
        redundancy_penalties = [0.0] * len(captions)
    else:
        e_c_batch = text_encoder(captions)
        # Normalize
        e_c_norm = e_c_batch / (np.linalg.norm(e_c_batch, axis=1, keepdims=True) + 1e-8)
        
        P_norm = prompt_embeddings / (
            np.linalg.norm(prompt_embeddings, axis=1, keepdims=True) + 1e-8
        )
        # sims: (num_prompts, num_captions) = (num_prompts, d) @ (num_captions, d).T
        # We want max similarity for each caption against all prompts
        # sims shape: (num_prompts, num_captions)
        sims = P_norm @ e_c_norm.T
        max_sims = np.max(sims, axis=0) # Shape (num_captions,)
        redundancy_penalties = (beta * max_sims).tolist()

    results = [l + r for l, r in zip(length_penalties, redundancy_penalties)]

    if is_batch:
        return results
    else:
        return results[0]


def description_length(
    prompt_set: PromptSet,
    prompt_embeddings: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.1,
) -> float:
    """
    DL(P) = sum_t C(c_t;P).
    """
    if len(prompt_set) == 0:
        return 0.0

    dl = 0.0
    for (_, c) in prompt_set:
        dl += caption_complexity(c, prompt_set, prompt_embeddings, alpha, beta)
    return dl


# ============================================================
# UNCERTAINTY VIA STOCHASTIC VLM CALLS (LOGIT-FREE)
# ============================================================

def get_stochastic_params(K: int) -> List[Tuple[float, float]]:
    """
    Generates K unique pairs of (temperature, top_p).
    Ensures that neither temperature nor top_p values repeat across the K sets.
    """
    params = []
    seen_t = set()
    seen_p = set()
    
    max_attempts = K * 50
    attempts = 0
    
    while len(params) < K and attempts < max_attempts:
        t = round(random.uniform(0.5, 1.5), 2)
        p = round(random.uniform(0.7, 1.0), 2)
        
        if t not in seen_t and p not in seen_p:
            seen_t.add(t)
            seen_p.add(p)
            params.append((t, p))
        
        attempts += 1
            
    while len(params) < K:
         t = round(random.uniform(0.5, 1.5), 2)
         p = round(random.uniform(0.7, 1.0), 2)
         params.append((t, p))
         
    return params

def _stochastic_label_samples(
    x: Union[Image, List[Image]],
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    K: int = 5,
    **gen_kwargs
) -> List[List[int]]:
    """
    Collect K stochastic predicted labels per image.
    Returns a list where each entry is [pred_1, ..., pred_K] for one image.
    """
    images = x if isinstance(x, list) else [x]
    counts_by_image = _stochastic_label_counts(
        x=images,
        system_prompt_1=system_prompt_1,
        system_prompt_2=system_prompt_2,
        prompt_set=prompt_set,
        K=K,
        **gen_kwargs,
    )
    image_predictions: List[List[int]] = []
    for counts in counts_by_image:
        preds: List[int] = []
        for label, count in sorted(counts.items(), key=lambda kv: int(kv[0])):
            preds.extend([int(label)] * int(count))
        image_predictions.append(preds)
    return image_predictions


def _stochastic_label_counts(
    x: Union[Image, List[Image]],
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    K: int = 5,
    **gen_kwargs
) -> List[Counter]:
    """
    Collect K stochastic predicted labels per image as label-count counters.
    Uses shared cache across selection methods.
    """
    images = x if isinstance(x, list) else [x]
    image_counts: List[Counter] = [Counter() for _ in images]
    remaining_calls = [int(K) for _ in images]

    cached_preds = {}
    if gen_kwargs.get("debug", False):
        cached_preds = load_cached_preds(gen_kwargs)

    cache_path = _resolve_uncertainty_cache_path(**gen_kwargs)
    cache_entries = _load_uncertainty_cache_entries(cache_path)
    cache_meta: List[Tuple[str, Optional[str], Dict[str, Any]]] = []
    cache_full_hit: List[bool] = [False for _ in images]

    for idx, img in enumerate(images):
        key, prompt_signature, key_payload = _uncertainty_cache_key(
            image_path=str(img),
            prompt_set=prompt_set,
            K=K,
            **gen_kwargs,
        )
        cache_meta.append((key, prompt_signature, key_payload))
        entry = cache_entries.get(key)
        if not isinstance(entry, dict):
            continue
        raw_counts = entry.get("label_counts", {})
        if not isinstance(raw_counts, dict):
            continue
        total_samples = int(entry.get("total_samples", 0) or 0)
        if total_samples <= 0:
            continue
        if total_samples > int(K):
            continue
        restored = Counter()
        for k_raw, v_raw in raw_counts.items():
            try:
                lbl = int(k_raw)
                cnt = int(v_raw)
                if cnt > 0:
                    restored[lbl] += cnt
            except (TypeError, ValueError):
                continue
        if int(sum(restored.values())) != total_samples:
            continue
        image_counts[idx] = restored
        remaining_calls[idx] = int(K) - total_samples
        if remaining_calls[idx] == 0:
            cache_full_hit[idx] = True

    max_steps = max(remaining_calls) if remaining_calls else 0
    for _ in range(max_steps):
        batch_to_query = []
        indices_to_query = []

        for idx, img in enumerate(images):
            if remaining_calls[idx] <= 0:
                continue
            if str(img) in cached_preds:
                image_counts[idx][int(cached_preds[str(img)]["label"])] += 1
                remaining_calls[idx] -= 1
            else:
                batch_to_query.append(img)
                indices_to_query.append(idx)

        if batch_to_query:
            results = vlm_query(
                batch_to_query, system_prompt_1, system_prompt_2, prompt_set,
                stochastic=True,
                **gen_kwargs
            )
            for idx, (y_hat, _) in zip(indices_to_query, results):
                image_counts[idx][int(y_hat)] += 1
                remaining_calls[idx] -= 1

    if any(rem != 0 for rem in remaining_calls):
        raise RuntimeError("Missing stochastic prediction for uncertainty estimation.")

    for idx, img in enumerate(images):
        counts = image_counts[idx]
        total = int(sum(counts.values()))
        if total != int(K):
            raise RuntimeError("Uncertainty cache/sample count mismatch.")
        if cache_full_hit[idx]:
            continue
        key, prompt_signature, key_payload = cache_meta[idx]
        probs = [float(v) / float(total) for v in counts.values() if int(v) > 0]
        entropy = float(-sum(p * np.log(p) for p in probs)) if probs else 0.0
        freq_max = max(counts.values()) if counts else 0
        uncertainty_val = float(1.0 - (float(freq_max) / float(total))) if total > 0 else 0.0

        entry = {
            "key": key,
            "dataset": key_payload.get("dataset", ""),
            "fold": key_payload.get("fold"),
            "round": int(key_payload.get("round", 0) or 0),
            "image_path": os.path.abspath(str(img)),
            "model": key_payload.get("model"),
            "temperature": key_payload.get("temperature"),
            "top_p": key_payload.get("top_p"),
            "max_tokens": key_payload.get("max_tokens"),
            "vlm_timeout_s": key_payload.get("vlm_timeout_s"),
            "K_uncertainty": int(K),
            "prompt_mode": key_payload.get("prompt_mode"),
            "prompt_signature": prompt_signature,
            "label_counts": {str(int(k)): int(v) for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))},
            "total_samples": int(total),
            "uncertainty": float(uncertainty_val),
            "entropy": float(entropy),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _append_uncertainty_cache_entry(cache_path, entry)

    return image_counts

def uncertainty(
    x: Union[Image, List[Image]],
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    K: int = 5,
    **gen_kwargs
) -> Union[float, List[float]]:
    """
    u(x;P) = 1 - max_y p_hat(y | x), using K stochastic calls.
    Supports batch processing.
    """
    is_batch = isinstance(x, list)
    image_counts = _stochastic_label_counts(
        x=x,
        system_prompt_1=system_prompt_1,
        system_prompt_2=system_prompt_2,
        prompt_set=prompt_set,
        K=K,
        **gen_kwargs,
    )

    uncertainties = []
    for counts in image_counts:
        total = int(sum(counts.values()))
        if total <= 0:
            uncertainties.append(0.0)
            continue
        freq_max = max(counts.values())
        uncertainties.append(1.0 - (float(freq_max) / float(total)))
        
    if is_batch:
        return uncertainties
    else:
        return uncertainties[0]


def entropy_uncertainty(
    x: Union[Image, List[Image]],
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    K: int = 5,
    **gen_kwargs
) -> Union[float, List[float]]:
    """
    u_ent(x;P) = H(p_hat(.|x)) using K stochastic calls, where:
    H(p) = -sum_k p_k log p_k.
    Supports batch processing.
    """
    is_batch = isinstance(x, list)
    image_counts = _stochastic_label_counts(
        x=x,
        system_prompt_1=system_prompt_1,
        system_prompt_2=system_prompt_2,
        prompt_set=prompt_set,
        K=K,
        **gen_kwargs,
    )

    entropy_scores: List[float] = []
    for counts in image_counts:
        total = int(sum(counts.values()))
        if total <= 0:
            entropy_scores.append(0.0)
            continue
        denom = float(total)
        probs = [float(v) / denom for v in counts.values() if v > 0]
        h = -sum(p * np.log(p) for p in probs)
        entropy_scores.append(float(h))

    if is_batch:
        return entropy_scores
    return entropy_scores[0]


# ============================================================
# EXPECTED CAPTION COMPLEXITY
# ============================================================

def expected_caption_complexity(
    x: Union[Image, List[Image]],
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    prompt_embeddings: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.1,
    **gen_kwargs
) -> Union[float, List[float]]:
    """
    \\hat C(x;P) = C(e_hat(x;P);P).
    Supports batch processing.
    """
    is_batch = isinstance(x, list)
    images = x if is_batch else [x]

    cached_preds = {}
    if gen_kwargs.get("debug", False):
        cached_preds = load_cached_preds(gen_kwargs)

    temp_results = [None] * len(images)
    batch_to_query = []
    indices_to_query = []

    for idx, img in enumerate(images):
        if str(img) in cached_preds:
            entry = cached_preds[str(img)]
            lbl = gen_kwargs.get('label_map')[entry['label']]
            cap = f"{entry['rationale']} C: {lbl}"
            temp_results[idx] = cap
        else:
            batch_to_query.append(img)
            indices_to_query.append(idx)

    if batch_to_query:
        results = vlm_query(
            batch_to_query, system_prompt_1, system_prompt_2, prompt_set,
            stochastic=False,
            **gen_kwargs
        )
        for idx, (label, e_hat) in zip(indices_to_query, results):
            lbl = gen_kwargs.get('label_map')[label]
            cap = f"{e_hat} C: {lbl}"
            temp_results[idx] = cap

    captions_hat = temp_results

    complexities = caption_complexity(
        captions_hat,
        prompt_set,
        prompt_embeddings,
        alpha=alpha,
        beta=beta,
    )

    if is_batch:
        return complexities
    else:
        return complexities[0]


# ============================================================
# SELECTION SCORE
# ============================================================

def selection_score(
    x: Union[Image, List[Image]],
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    prompt_embeddings: np.ndarray,
    lambda_c: float = 0.5,
    K: int = 5,
    alpha: float = 0.01,
    beta: float = 0.1,
    **gen_kwargs
) -> Union[float, List[float]]:
    """
    s(x;P) = u(x;P) - lambda_c * \\hat C(x;P).
    Supports batch processing.
    """
    is_batch = isinstance(x, list)
    
    u_x = uncertainty(x, system_prompt_1, system_prompt_2, prompt_set, K=K, **gen_kwargs)
    c_x = expected_caption_complexity(
        x, system_prompt_1, system_prompt_2, prompt_set, prompt_embeddings,
        alpha=alpha, beta=beta, **gen_kwargs
    )
    
    if is_batch:
        return [u - lambda_c * c for u, c in zip(u_x, c_x)]
    else:
        return u_x - lambda_c * c_x


# ============================================================
# MDL LOSS
# ============================================================

def zero_one_error(preds: List[int], labels: List[int]) -> float:
    errors = sum(int(p != y) for p, y in zip(preds, labels))
    return errors / len(labels)


def mdl_loss(
    prompt_set: PromptSet,
    val_data: List[LabeledExample],
    system_prompt_1: str,
    system_prompt_2_template: str,
    alpha: float,
    beta: float,
    lambda_mdl: float,
    val_batch_size: int = 5,
    **gen_kwargs
) -> Tuple[float, List[int], List[int]]:
    """
    L_lambda(P) = error(P on validation) + lambda * DL(P).
    Returns (loss, preds, labels)
    """
    # Caption embeddings
    captions = [c for (_, c) in prompt_set]
    if len(captions) > 0:
        prompt_embeddings = text_encoder(captions)
    else:
        prompt_embeddings = np.zeros((0, 1), dtype=float)

    # Load cached predictions if log path is provided and debug is True
    cached_preds = {}
    if gen_kwargs.get("debug", False):
        cached_preds = load_cached_preds(gen_kwargs)

    preds, labels = [], []
    
    # Process validation data in batches
    for i in tqdm(range(0, len(val_data), val_batch_size), desc="Validation"):
        batch = val_data[i:i + val_batch_size]
        batch_x = [x for x, _ in batch]
        batch_y = [y for _, y in batch]
        
        # Check cache for each item in batch
        # If all in cache, use cache. If mixed, query for missing? 
        # For simplicity, if any missing, we might query all or handle individually.
        # But to leverage batching, we should group non-cached items.
        
        # Let's split into cached and non-cached
        non_cached_indices = []
        non_cached_images = []
        
        batch_preds = [None] * len(batch)
        
        for idx, x in enumerate(batch_x):
            if str(x) in cached_preds:
                batch_preds[idx] = cached_preds[str(x)]["label"]
            else:
                non_cached_indices.append(idx)
                non_cached_images.append(x)
        
        if non_cached_images:
            # Query VLM for non-cached images
            # vlm_query returns list of (label, rationale)
            results = vlm_query(
                non_cached_images, system_prompt_1, system_prompt_2_template, prompt_set,
                stochastic=False,
                **gen_kwargs
            )
            
            # Fill in predictions
            for idx, (y_hat, _) in zip(non_cached_indices, results):
                batch_preds[idx] = y_hat
                
        # Append to main lists
        for p, y in zip(batch_preds, batch_y):
            preds.append(int(p))
            labels.append(int(y))

    err = zero_one_error(preds, labels)
    dl = description_length(prompt_set, prompt_embeddings, alpha, beta)

    return err + lambda_mdl * dl, preds, labels


# ============================================================
# APT–MDL CLASS
# ============================================================

class APTMDL:
    def __init__(
        self,
        system_prompt_1_path: str,
        system_prompt_2_path: str,
        selection_method: str,
        alpha: float = 0.01,
        beta: float = 0.1,
        lambda_mdl: float = 0.1,
        lambda_c: float = 0.5,
        K_uncertainty: int = 5,
        mdl_tol: float = 1e-3,
        max_rounds: int = 20,
        candidate_pool_size: int = -1,
        dts_k: int = 60,
        dts_k_rho: int = 30,
        dts_k_t: int = 20,
        dts_k_b: int = 15,
        dts_mutual_knn: bool = False,
        dts_mcluster_min: int = 20,
        dts_c_tiny: int = 1,
        dts_max_per_basin: int = 2,
        dts_deg_min_tiny: int = 10,
        dts_b_min_tiny: float = 0.6,
        dts_tune_hparams: bool = True,
        dts_clip_model_name: str = DEFAULT_DTS_CLIP_MODEL_NAME,
        diagnostic_mode: bool = False,
        show_interactive: bool = False,
        diagnostic_every: int = 2,
        diagnostic_outdir: str = "diagnostics",
        diagnostic_seed: int = 0,
        max_images_per_panel: int = 40,
        clip_batch_size: int = 32,
        val_batch_size: int = 5,
        debug: bool = False,
        oracle_path: str = None,
        prompt_set_path: str = "final_prompt_set.json",
        logs_dir: str = "logs",
        fold: int = None,
        stopping_accuracy: float = 90.0,
        active_set_batch_size: Optional[int] = None,
    ):
        self.S_1 = load_files(system_prompt_1_path)
        # System prompt 2 is now a template string handled in code, not loaded from file
        self.S_2_template = "Assume that you are forced to classify the {N} images below based on the treatment without any further information. Now use the information analyzed from above images to provide the rationale and classification for the below image. You should focus on six features for your classification: cellular organization, layering pattern, purkinje cells, granule cell layer, overall structure and staining pattern as described above. Your rationale should include descriptions of all six features. your response should be given in the exact format for each image. 'R:' should indicate the textual explanation of the image based on the features described above. 'C:' should indicate the classification based on your rationale - either 'lurcher' for Lurcher mutant group or 'wild' for wild-type/normal. Be concise and specific. Do not include anything else in the output."
        self.alpha = alpha
        self.beta = beta
        self.lambda_mdl = lambda_mdl
        self.lambda_c = lambda_c
        self.K_uncertainty = K_uncertainty
        # Explicit strategy switch:
        # "mdl" keeps legacy MDL score, "entropy" uses Shannon entropy, "dts" uses CLIP+density-tree.
        self.selection_method = selection_method.lower()
        if self.selection_method not in ("mdl", "entropy", "dts"):
            raise ValueError(f"Unsupported selection_method: {selection_method}")
        self.mdl_tol = mdl_tol
        self.max_rounds = max_rounds
        self.candidate_pool_size = candidate_pool_size
        self.dts_k = dts_k
        self.dts_k_rho = dts_k_rho
        self.dts_k_t = dts_k_t
        self.dts_k_b = dts_k_b
        self.dts_mutual_knn = dts_mutual_knn
        self.dts_mcluster_min = dts_mcluster_min
        self.dts_c_tiny = dts_c_tiny
        self.dts_max_per_basin = dts_max_per_basin
        self.dts_deg_min_tiny = dts_deg_min_tiny
        self.dts_b_min_tiny = dts_b_min_tiny
        self.dts_tune_hparams = bool(dts_tune_hparams)
        self.dts_clip_model_name = _resolve_dts_clip_model_name(dts_clip_model_name)
        self.selection_artifact_token = _selection_artifact_token(
            selection_method=self.selection_method,
            dts_clip_model_name=self.dts_clip_model_name,
            active_set_batch_size=active_set_batch_size,
            candidate_pool_size=self.candidate_pool_size,
        )
        self.dts_tuner_state = {
            "overmerged_streak": 0,
            "fragmented_streak": 0,
            "boundary_flat_streak": 0,
            "cooldown_k": 0,
            "freeze_tuning_rounds": 0,
            "prev_singleton_frac": None,
            "prev_basin_median_size": None,
            "prev_num_deg0": None,
            "prev_hparams": None,
        }
        self.diagnostic_mode = bool(diagnostic_mode)
        self.show_interactive = bool(show_interactive)
        self.diagnostic_every = max(1, int(diagnostic_every))
        self.diagnostic_outdir = str(diagnostic_outdir)
        self.diagnostic_seed = int(diagnostic_seed)
        self.max_images_per_panel = max(1, int(max_images_per_panel))
        self.clip_batch_size = clip_batch_size
        self.val_batch_size = val_batch_size
        self.debug = debug
        self.oracle_path = oracle_path
        self.stopping_accuracy = stopping_accuracy
        self.last_oracle_accuracy = None
        self.last_validation_avg_class_accuracy = None
        self.last_active_set_metrics = None
        self.initial_prompt_items: List[Dict[str, Any]] = []
        self.initial_global_metrics: Dict[str, Any] = {}
        # Keep prompt-set outputs method-specific by default.
        self.prompt_set_path = with_selection_suffix(prompt_set_path, self.selection_artifact_token)
        self.logs_dir = logs_dir
        self.fold = fold

        self.prompt_set: PromptSet = []

    # -------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------
    def evaluate(self, val_data: List[LabeledExample], round_num: int, **gen_kwargs) -> float:
        """
        Calculates MDL loss, accuracy, and logs results.
        Returns the MDL loss.
        """
        print("Performing evaluation for round {}.".format(round_num))
        # Call mdl_loss to get loss and predictions
        loss, preds, labels = mdl_loss(
            self.prompt_set,
            val_data,
            self.S_1,
            self.S_2_template,
            alpha=self.alpha,
            beta=self.beta,
            lambda_mdl=self.lambda_mdl,
            val_batch_size=self.val_batch_size,
            **gen_kwargs
        )

        # Calculate metrics
        correct = sum(1 for p, y in zip(preds, labels) if p == y)
        total = len(labels)
        accuracy = correct / total if total > 0 else 0.0

        # Class-level accuracy and confusion matrix
        # Assuming binary classification for now based on label_map default, but making it general
        # Get unique labels from data and predictions
        unique_labels = sorted(list(set(labels) | set(preds)))
        
        class_accuracies = {}
        confusion_matrix = {l: {l2: 0 for l2 in unique_labels} for l in unique_labels}
        
        for p, y in zip(preds, labels):
            confusion_matrix[y][p] += 1
            
        for l in unique_labels:
            total_l = sum(confusion_matrix[l].values())
            correct_l = confusion_matrix[l][l]
            class_accuracies[l] = correct_l / total_l if total_l > 0 else 0.0
            
        avg_class_accuracy = sum(class_accuracies.values()) / len(class_accuracies) if class_accuracies else 0.0

        # Log results
        # Construct filename
        dataset_name = gen_kwargs.get("dataset", "unknown")
        K = self.K_uncertainty
        max_rounds = self.max_rounds
        pool_size = "all" if self.candidate_pool_size in (None, -1) else self.candidate_pool_size
        init_batch = gen_kwargs.get("initial_batch_size", 10)
        temp = gen_kwargs.get("temperature", 0.7)
        
        log_filename = f"{dataset_name}_selection={self.selection_artifact_token}_K_{K}_rounds_{max_rounds}_pool_{pool_size}_batch_{init_batch}_temp={temp}.log"
        if self.fold is not None:
             log_filename = f"{dataset_name}_fold={self.fold}_selection={self.selection_artifact_token}_K={K}_rounds={max_rounds}_pool={pool_size}_batch={init_batch}_temp={temp}.log"
        log_path = os.path.join(self.logs_dir, log_filename)

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            
        log_entry = {
            "round": round_num,
            "mdl_loss": loss,
            "accuracy": accuracy,
            "avg_class_accuracy": avg_class_accuracy,
            "class_accuracies": class_accuracies,
            "confusion_matrix": confusion_matrix,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # Append to log file (JSON lines format or just append to a list in JSON)
            # Using JSON lines for easier appending
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(f"Evaluation results logged to {log_path}")
        except Exception as e:
            print(f"Error logging evaluation results: {e}")
            
        print(f"Round {round_num} Eval: Loss={loss:.4f}, Acc={accuracy:.4f}, AvgClassAcc={avg_class_accuracy:.4f}")
        
        return loss

    # -------------------------------------------------------
    # ORACLE (EXPERT COMMENTS/ EDITS)
    # -------------------------------------------------------
    def oracle_label_and_edit(self, x: Union[Image, List[Image]], predictions: List[Tuple[Label, str]] = None, **gen_kwargs) -> Union[Tuple[Label, Caption], List[Tuple[Label, Caption]]]:
        """
        Oracle correction integration via APT-v3/oracle.py.
        Persists round prompts before/after correction and returns corrected (label_idx, caption).
        """
        is_batch = isinstance(x, list)
        images = x if is_batch else [x]
        label_map = gen_kwargs.get("label_map", ["wild", "lurcher"])
        label_lookup = {str(lbl).lower(): str(lbl) for lbl in label_map}

        # Never consume externally preserved predictions outside debug mode.
        if not self.debug:
            predictions = None

        # 1) Predictions used to build pre-correction prompt file.
        if predictions is None:
            # Query VLM in chunks to avoid sending all active-set images in one request.
            vlm_query_batch_size = int(
                gen_kwargs.get("vlm_query_batch_size", gen_kwargs.get("selection_batch_size", 5)) or 5
            )
            if vlm_query_batch_size <= 0:
                vlm_query_batch_size = 5

            predictions = []
            for i in range(0, len(images), vlm_query_batch_size):
                batch_images = images[i:i + vlm_query_batch_size]
                batch_predictions = vlm_query(
                    batch_images, self.S_1, self.S_2_template, self.prompt_set,
                    stochastic=False,
                    **gen_kwargs
                )
                if isinstance(batch_predictions, tuple):
                    batch_predictions = [batch_predictions]
                predictions.extend(batch_predictions)
        if len(predictions) != len(images):
            raise ValueError("Length mismatch between images and predictions in oracle_label_and_edit.")

        # 2) Persist pre-correction prompts.
        dataset_name = gen_kwargs.get("dataset", "unknown")
        fold = self.fold if self.fold is not None else gen_kwargs.get("fold")
        round_num = gen_kwargs.get("round_num", 0)
        prompts_root = gen_kwargs.get("prompts_root", "prompt_sets")
        prompts_in_path, prompts_out_path = build_round_prompt_paths(
            prompts_root=prompts_root,
            dataset_name=dataset_name,
            fold=fold,
            round_num=round_num,
            selection_method=self.selection_artifact_token,
        )

        prompt_items = []
        pre_by_path = {}
        pre_label_idx_by_path = {}
        for img, (pred_label_idx, pred_rationale) in zip(images, predictions):
            abs_img = os.path.abspath(str(img))
            label_idx = int(pred_label_idx) if isinstance(pred_label_idx, Integral) else 0
            if not (0 <= label_idx < len(label_map)):
                label_idx = 0
            label_str = str(label_map[label_idx])
            rationale = "" if pred_rationale is None else str(pred_rationale)
            prompt_items.append({
                "image_path": abs_img,
                "class": label_str,
                # Keep both keys to stay compatible with APT-v3/oracle.py and current MDL naming.
                "rationale": rationale,
                "explanation": rationale
            })
            pre_by_path[abs_img] = (label_str, rationale)
            pre_label_idx_by_path[abs_img] = int(label_idx)

        payload = {"prompts": prompt_items}
        with open(prompts_in_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"Saved pre-correction prompts: {prompts_in_path}")

        # 3) Run APT-v3 oracle correction (or passthrough in debug mode).
        if self.debug:
            with open(prompts_out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            print(f"Debug mode: wrote passthrough corrected prompts: {prompts_out_path}")
        else:
            default_oracle_script_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "APT-v3", "oracle.py")
            )
            oracle_script_path = gen_kwargs.get("oracle_script_path") or default_oracle_script_path
            oracle_dataset_name = gen_kwargs.get("oracle_dataset_name") or resolve_oracle_dataset_name(
                dataset_name, label_map
            )

            cmd = [
                sys.executable,
                oracle_script_path,
                "--input_file", prompts_in_path,
                "--dataset", oracle_dataset_name,
                "--output_file", prompts_out_path,
            ]
            if self.oracle_path:
                cmd.extend(["--prompt_bank_path", self.oracle_path])
            print(f"Running oracle correction: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running oracle script: {e}")
                raise e

        if not os.path.exists(prompts_out_path):
            raise FileNotFoundError(f"Corrected prompts file not found: {prompts_out_path}")

        with open(prompts_out_path, "r", encoding="utf-8") as f:
            corrected_payload = json.load(f)
        oracle_accuracy = corrected_payload.get("accuracy")
        try:
            self.last_oracle_accuracy = float(oracle_accuracy) if oracle_accuracy is not None else None
        except (TypeError, ValueError):
            self.last_oracle_accuracy = None
        if self.last_oracle_accuracy is not None:
            print(f"Oracle active-set accuracy (round {round_num}): {self.last_oracle_accuracy:.1f}%")
        corrected_items = corrected_payload.get("prompts", [])
        if not isinstance(corrected_items, list):
            raise ValueError("Oracle output must contain a list under 'prompts'.")

        # 4) Map corrected entries back to images and update cache.
        corrected_by_path = {}
        manual_corrected_paths = set()
        for item in corrected_items:
            img_path = item.get("image_path") or item.get("path")
            if not img_path:
                continue
            abs_img = os.path.abspath(str(img_path))

            cls_raw = str(item.get("class", item.get("c", label_map[0]))).strip()
            cls = label_lookup.get(cls_raw.lower(), str(label_map[0]))
            rationale = item.get("rationale")
            if rationale is None:
                rationale = item.get("explanation", item.get("e", ""))
            corrected_by_path[abs_img] = (cls, "" if rationale is None else str(rationale))
            if bool(item.get("manual_corrected", False)):
                manual_corrected_paths.add(abs_img)

        active_paths: List[str] = []
        active_y_true: List[int] = []
        active_y_pred: List[Optional[int]] = []
        for img in images:
            abs_img = os.path.abspath(str(img))
            true_cls, _ = corrected_by_path.get(abs_img, pre_by_path.get(abs_img, (str(label_map[0]), "")))
            true_idx = label_map.index(true_cls) if true_cls in label_map else 0
            pred_idx = pre_label_idx_by_path.get(abs_img)
            active_paths.append(abs_img)
            active_y_true.append(int(true_idx))
            active_y_pred.append(int(pred_idx) if pred_idx is not None else None)

        active_class_metrics = self._summarize_class_metrics(
            image_paths=active_paths,
            y_true_labels=active_y_true,
            y_pred_labels=active_y_pred,
            label_map=list(label_map),
        )
        self.last_active_set_metrics = {
            "round": int(round_num),
            "active_set_size": int(len(active_paths)),
            "class_totals": active_class_metrics["class_totals"],
            "class_correct": active_class_metrics["class_correct"],
            "case_ids_by_class": active_class_metrics["case_ids_by_class"],
            "case_ids_correct_by_class": active_class_metrics["case_ids_correct_by_class"],
            "class_accuracy_pct": active_class_metrics["class_accuracy_pct"],
            "avg_class_accuracy_pct": active_class_metrics["avg_class_accuracy_pct"],
            "oracle_accuracy_pct": float(self.last_oracle_accuracy) if self.last_oracle_accuracy is not None else None,
            "prompts_in_path": prompts_in_path,
            "prompts_out_path": prompts_out_path,
        }
        try:
            global_prompts_path = self._rebuild_global_prompts_file(
                prompts_root=str(prompts_root),
                dataset_name=str(dataset_name),
                fold=fold,
                selection_method=self.selection_artifact_token,
                label_map=list(label_map or []),
            )
            if global_prompts_path:
                self.last_active_set_metrics["global_prompts_path"] = str(global_prompts_path)
        except Exception as e:
            print(f"Warning: failed to rebuild global prompts file: {e}")

        final_results = []
        new_cache_entries = []
        for img in images:
            abs_img = os.path.abspath(str(img))
            cls, rationale = corrected_by_path.get(abs_img, pre_by_path.get(abs_img, (str(label_map[0]), "")))
            label_idx = label_map.index(cls) if cls in label_map else 0
            caption = f"{rationale} C: {cls}"
            final_results.append((label_idx, caption))
            if abs_img in manual_corrected_paths:
                new_cache_entries.append({
                    "rationale": rationale,
                    "class": cls,
                    "image_path": abs_img
                })

        if self.oracle_path:
            try:
                if not new_cache_entries:
                    print(
                        f"No manually corrected entries in round {round_num}; "
                        f"skipping prompt-bank update at {self.oracle_path}."
                    )
                    if is_batch:
                        return final_results
                    return final_results[0]

                current_oracle_data = []
                if os.path.exists(self.oracle_path):
                    with open(self.oracle_path, "r", encoding="utf-8") as f:
                        current_oracle_data = json.load(f)

                updated_oracle_data = None

                # Format-aware merge:
                # 1) Prompt-bank format: {"prompts":[{"path","class","explanation"}, ...]}
                # 2) Cache format: [{"image_path","class","rationale"}, ...]
                if isinstance(current_oracle_data, dict) and isinstance(current_oracle_data.get("prompts"), list):
                    bank_items = current_oracle_data.get("prompts", [])
                    merged = {}
                    for item in bank_items:
                        if not isinstance(item, dict):
                            continue
                        path_key = item.get("path") or item.get("image_path")
                        if path_key:
                            rel_key = to_class_rooted_relative_path(str(path_key), label_map)
                            item_copy = dict(item)
                            item_copy["path"] = rel_key
                            merged[rel_key] = item_copy

                    for entry in new_cache_entries:
                        p = to_class_rooted_relative_path(str(entry["image_path"]), label_map)
                        existing = merged.get(p, {})
                        merged[p] = {
                            "path": p,
                            "class": entry["class"],
                            "explanation": entry["rationale"],
                            **{k: v for k, v in existing.items() if k not in {"path", "class", "explanation"}},
                        }

                    current_oracle_data["prompts"] = list(merged.values())
                    updated_oracle_data = current_oracle_data

                elif isinstance(current_oracle_data, list):
                    merged = {}
                    for item in current_oracle_data:
                        if not isinstance(item, dict):
                            continue
                        path_key = item.get("image_path") or item.get("path")
                        if path_key:
                            merged[str(path_key)] = item

                    for entry in new_cache_entries:
                        merged[str(entry["image_path"])] = entry
                    updated_oracle_data = list(merged.values())

                else:
                    # Unknown/empty format: fallback to cache list format.
                    updated_oracle_data = list(new_cache_entries)

                oracle_dir = os.path.dirname(self.oracle_path)
                if oracle_dir and not os.path.exists(oracle_dir):
                    os.makedirs(oracle_dir, exist_ok=True)
                with open(self.oracle_path, "w", encoding="utf-8") as f:
                    json.dump(updated_oracle_data, f, indent=4)
                print(f"Updated oracle cache with {len(new_cache_entries)} entries from round {round_num}.")
            except Exception as e:
                print(f"Error updating oracle cache: {e}")

        if is_batch:
            return final_results
        return final_results[0]

    # -------------------------------------------------------
    # CHECKPOINTING
    # -------------------------------------------------------
    def save_checkpoint(self, round_num: int, validation_avg_class_accuracy: Optional[float], checkpoint_path: str):
        """
        Saves the current state to a JSON file.
        """
        # Convert prompt_set and unlabeled_data to serializable format
        # prompt_set is saved as a list of dictionaries: {"image_path": path, "caption": caption}
        # unlabeled_data is saved as a list of tuples: (image_path, label)

        round_state = {
            "round": round_num,
            "validation_avg_class_accuracy": validation_avg_class_accuracy,
            "selection_method": self.selection_method,
            "prompt_set": [{"image_path": str(x), "caption": c} for x, c in self.prompt_set],
            "unlabeled_data": [(str(x), l) for x, l in self.unlabeled_data]
        }
        if self.selection_method == "dts":
            round_state["dts_hyperparameters"] = {
                "k": int(self.dts_k),
                "k_rho": int(self.dts_k_rho),
                "k_t": int(self.dts_k_t),
                "k_b": int(self.dts_k_b),
                "mcluster_min": int(self.dts_mcluster_min),
                "c_tiny": int(self.dts_c_tiny),
                "max_per_basin": int(self.dts_max_per_basin),
                "deg_min_tiny": int(self.dts_deg_min_tiny),
                "b_min_tiny": float(self.dts_b_min_tiny),
                "tune_hparams": bool(self.dts_tune_hparams),
            }
            round_state["dts_tuner_state"] = {
                "overmerged_streak": int(self.dts_tuner_state.get("overmerged_streak", 0)),
                "fragmented_streak": int(self.dts_tuner_state.get("fragmented_streak", 0)),
                "boundary_flat_streak": int(self.dts_tuner_state.get("boundary_flat_streak", 0)),
                "cooldown_k": int(self.dts_tuner_state.get("cooldown_k", 0)),
                "freeze_tuning_rounds": int(self.dts_tuner_state.get("freeze_tuning_rounds", 0)),
                "prev_singleton_frac": self.dts_tuner_state.get("prev_singleton_frac"),
                "prev_basin_median_size": self.dts_tuner_state.get("prev_basin_median_size"),
                "prev_num_deg0": self.dts_tuner_state.get("prev_num_deg0"),
                "prev_hparams": self.dts_tuner_state.get("prev_hparams"),
            }
        
        try:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            existing_state: Dict[str, Any] = {}
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        existing_state = loaded
                except Exception:
                    existing_state = {}

            round_history: Dict[str, Any] = {}
            if isinstance(existing_state.get("round_history"), dict):
                round_history = dict(existing_state["round_history"])
            elif "round" in existing_state and "prompt_set" in existing_state and "unlabeled_data" in existing_state:
                # Migrate legacy single-snapshot checkpoint into history.
                try:
                    legacy_round = int(existing_state.get("round"))
                    round_history[str(legacy_round)] = {
                        k: existing_state.get(k)
                        for k in (
                            "round",
                            "validation_avg_class_accuracy",
                            "selection_method",
                            "prompt_set",
                            "unlabeled_data",
                            "dts_hyperparameters",
                            "dts_tuner_state",
                        )
                        if k in existing_state
                    }
                except Exception:
                    pass

            now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            round_entry = dict(round_state)
            round_entry["saved_at"] = now_ts
            round_history[str(int(round_num))] = round_entry

            state = dict(round_state)
            state["created_at"] = existing_state.get("created_at", now_ts)
            state["updated_at"] = now_ts
            state["latest_round"] = int(round_num)
            state["round_history"] = round_history

            with open(checkpoint_path, 'w') as f:
                json.dump(state, f, indent=4)
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, Optional[float]]:
        """
        Loads state from a checkpoint file.
        Returns (last_round, last_validation_avg_class_accuracy).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)

            # Backward compatibility: old checkpoints without this field are MDL.
            checkpoint_method = state.get("selection_method", "mdl")
            if checkpoint_method != self.selection_method:
                raise ValueError(
                    f"Checkpoint selection_method={checkpoint_method} does not match "
                    f"current selection_method={self.selection_method}."
                )

            state_for_resume = state
            round_history = state.get("round_history")
            if isinstance(round_history, dict) and len(round_history) > 0:
                latest_round_key = None
                latest_round_num = None
                try:
                    candidate_keys = [k for k in round_history.keys() if str(k).strip() != ""]
                    if candidate_keys:
                        if state.get("latest_round") is not None:
                            preferred = str(int(state.get("latest_round")))
                            if preferred in round_history:
                                latest_round_key = preferred
                                latest_round_num = int(preferred)
                        if latest_round_key is None:
                            latest_round_num = max(int(k) for k in candidate_keys)
                            latest_round_key = str(latest_round_num)
                except Exception:
                    latest_round_key = sorted(round_history.keys())[-1]
                    try:
                        latest_round_num = int(latest_round_key)
                    except Exception:
                        latest_round_num = None

                if latest_round_key is not None and isinstance(round_history.get(latest_round_key), dict):
                    state_for_resume = round_history[latest_round_key]
                    if latest_round_num is not None:
                        state_for_resume = dict(state_for_resume)
                        state_for_resume.setdefault("round", int(latest_round_num))

            # Load prompt_set from list of dictionaries
            prompt_state = state_for_resume.get("prompt_set", state.get("prompt_set", []))
            unlabeled_state = state_for_resume.get("unlabeled_data", state.get("unlabeled_data", []))
            self.prompt_set = [(item["image_path"], item["caption"]) for item in prompt_state]
            self.unlabeled_data = [(x, l) for x, l in unlabeled_state]

            # Backward compatibility: if old checkpoint has mdl_loss only, return that fallback value.
            last_validation_avg_class_accuracy = state_for_resume.get(
                "validation_avg_class_accuracy",
                state.get("validation_avg_class_accuracy"),
            )
            if last_validation_avg_class_accuracy is None:
                # Backward compatibility with older checkpoints.
                last_validation_avg_class_accuracy = state_for_resume.get(
                    "active_set_accuracy",
                    state.get("active_set_accuracy"),
                )
            if last_validation_avg_class_accuracy is None and "mdl_loss" in state:
                try:
                    last_validation_avg_class_accuracy = float(state["mdl_loss"])
                except (TypeError, ValueError):
                    last_validation_avg_class_accuracy = None

            if self.selection_method == "dts":
                dts_hparams = state_for_resume.get(
                    "dts_hyperparameters",
                    state.get("dts_hyperparameters", {}),
                )
                self.dts_k = int(dts_hparams.get("k", self.dts_k))
                self.dts_k_rho = int(dts_hparams.get("k_rho", self.dts_k_rho))
                self.dts_k_t = int(dts_hparams.get("k_t", self.dts_k_t))
                self.dts_k_b = int(dts_hparams.get("k_b", self.dts_k_b))
                self.dts_mcluster_min = int(dts_hparams.get("mcluster_min", self.dts_mcluster_min))
                self.dts_c_tiny = int(dts_hparams.get("c_tiny", self.dts_c_tiny))
                self.dts_max_per_basin = int(dts_hparams.get("max_per_basin", self.dts_max_per_basin))
                self.dts_deg_min_tiny = int(dts_hparams.get("deg_min_tiny", self.dts_deg_min_tiny))
                self.dts_b_min_tiny = float(dts_hparams.get("b_min_tiny", self.dts_b_min_tiny))
                saved_state = state_for_resume.get(
                    "dts_tuner_state",
                    state.get("dts_tuner_state", {}),
                )
                self.dts_tuner_state = {
                    "overmerged_streak": int(saved_state.get("overmerged_streak", 0)),
                    "fragmented_streak": int(saved_state.get("fragmented_streak", 0)),
                    "boundary_flat_streak": int(saved_state.get("boundary_flat_streak", 0)),
                    "cooldown_k": int(saved_state.get("cooldown_k", 0)),
                    "freeze_tuning_rounds": int(saved_state.get("freeze_tuning_rounds", 0)),
                    "prev_singleton_frac": saved_state.get("prev_singleton_frac"),
                    "prev_basin_median_size": saved_state.get("prev_basin_median_size"),
                    "prev_num_deg0": saved_state.get("prev_num_deg0"),
                    "prev_hparams": saved_state.get("prev_hparams"),
                }

            last_round = int(state_for_resume.get("round", state.get("round", 0)))
            print(f"Checkpoint loaded from {checkpoint_path}. Resuming from round {last_round + 1}")
            return last_round, last_validation_avg_class_accuracy
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e

    # -------------------------------------------------------
    # INITIAL SEED
    # -------------------------------------------------------
    def initialize_seed(self, init_prompts_path: str, dataset: str, fold: int = None):
        """
        Load seed examples from a JSON file.
        """

        if fold is not None:
            file_path = os.path.join(init_prompts_path, f"{dataset}_fold={fold}")
        else:
            file_path = os.path.join(init_prompts_path, dataset)
        
        # Check if file exists, if not try with .json extension
        if not os.path.exists(file_path) and os.path.exists(file_path + ".json"):
            file_path += ".json"
            
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Initial prompts file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)
            

        seed_examples: List[Tuple[str, str]] = []
        seed_items: List[Dict[str, Any]] = []

        source_items = []
        if isinstance(data, dict):
            if isinstance(data.get("prompt"), list):
                source_items = data.get("prompt", [])
            elif isinstance(data.get("prompts"), list):
                source_items = data.get("prompts", [])

        for raw_item in source_items:
            if not isinstance(raw_item, dict):
                continue
            image_path = str(raw_item.get("image_path", "")).strip()
            if not image_path:
                continue
            rationale = raw_item.get("rationale")
            if rationale is None:
                rationale = raw_item.get("explanation", "")
            rationale = "" if rationale is None else str(rationale)

            label = str(raw_item.get("class", "")).strip()
            if not label and isinstance(raw_item.get("caption"), str):
                cap_text = str(raw_item.get("caption"))
                m = re.search(r"[cC]\s*:\s*([^\n\r]+)", cap_text)
                if m:
                    label = m.group(1).strip().split()[0].strip(".,;:*\"'`_[]{}()")

            caption = f"{rationale} C: {label}".strip() if label else rationale
            seed_examples.append((image_path, caption))
            seed_items.append({
                "image_path": image_path,
                "class": label,
                "rationale": rationale,
                "explanation": rationale,
                "manual_corrected": bool(raw_item.get("manual_corrected", False)),
            })

        self.prompt_set = list(seed_examples)
        self.initial_prompt_items = list(seed_items)
        init_acc = None
        if isinstance(data, dict) and "accuracy" in data and data.get("accuracy") not in (None, ""):
            try:
                init_acc = float(data.get("accuracy"))
            except (TypeError, ValueError):
                init_acc = None
        self.initial_global_metrics = {
            "caption correction count": int(self._to_int(data.get("caption correction count", 0), default=0)) if isinstance(data, dict) else 0,
            "Duration in minutes": float(self._to_float(data.get("Duration in minutes", 0.0), default=0.0)) if isinstance(data, dict) else 0.0,
            "accuracy": init_acc,
        }

    def _caption_to_label_index(self, caption: str, label_map: Optional[List[str]]) -> Optional[int]:
        if caption is None or not label_map:
            return None
        text = str(caption)
        match = re.search(r"[cC]\s*:\s*([^\n\r]+)", text)
        if not match:
            return None
        rhs = match.group(1).strip().lstrip("*'\"`_- ")
        token = ""
        for part in rhs.split():
            cleaned = part.strip(".,;:*\"'`_[]{}()")
            if cleaned:
                token = cleaned
                break
        if not token:
            return None
        labels_lower = [str(x).lower() for x in label_map]
        token_lower = token.lower()
        if token_lower in labels_lower:
            return labels_lower.index(token_lower)
        return None

    def _load_candidate_predicted_labels(
        self,
        candidate_paths: List[str],
        vlm_log_path: Optional[str],
        label_map: Optional[List[str]],
    ) -> Optional[List[Any]]:
        if not vlm_log_path or not os.path.exists(vlm_log_path):
            return None
        try:
            with open(vlm_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            if not isinstance(log_data, list):
                return None
        except Exception:
            return None

        by_path: Dict[str, Any] = {}
        labels_lower = [str(x).lower() for x in (label_map or [])]
        for entry in log_data:
            if not isinstance(entry, dict):
                continue
            path_key = entry.get("image_path")
            if path_key is None:
                continue
            label_val = entry.get("label")
            norm_label: Any = None
            if isinstance(label_val, Integral):
                norm_label = int(label_val)
            elif isinstance(label_val, str):
                token = label_val.strip().lower()
                if labels_lower and token in labels_lower:
                    norm_label = labels_lower.index(token)
                else:
                    norm_label = label_val
            by_path[os.path.abspath(str(path_key))] = norm_label

        output = []
        has_any = False
        for p in candidate_paths:
            v = by_path.get(os.path.abspath(str(p)))
            output.append(v)
            if v is not None:
                has_any = True
        return output if has_any else None

    def _infer_label_from_path(self, image_path: str, label_map: Optional[List[str]]) -> Optional[int]:
        if not image_path or not label_map:
            return None
        labels_lower = [str(lbl).strip().lower() for lbl in label_map]
        # Match only directory names (not filename tokens), case-insensitive.
        dir_parts = [p.strip().lower() for p in Path(str(image_path)).parent.parts if p and p != "."]
        for part in dir_parts:
            if part in labels_lower:
                return int(labels_lower.index(part))
        return None

    def _normalize_prediction_label(self, pred: Any, label_map: Optional[List[str]]) -> Optional[int]:
        if pred is None or not label_map:
            return None
        labels_lower = [str(lbl).strip().lower() for lbl in label_map]
        if isinstance(pred, Integral):
            idx = int(pred)
            return idx if 0 <= idx < len(labels_lower) else None
        token = str(pred).strip().lower()
        if token in labels_lower:
            return labels_lower.index(token)
        if token.startswith("c:"):
            token = token[2:].strip()
            if token in labels_lower:
                return labels_lower.index(token)
        return None

    def _extract_case_id_from_path(self, image_path: str, label_map: Optional[List[str]]) -> str:
        raw = str(image_path or "").strip()
        if not raw:
            return ""
        parts = [p for p in Path(raw).parts if p and p != "."]
        labels_lower = [str(lbl).strip().lower() for lbl in (label_map or [])]
        lowered_parts = [str(p).strip().lower() for p in parts]
        for i, part in enumerate(lowered_parts):
            if part in labels_lower and i + 1 < len(parts):
                return str(parts[i + 1])
        if len(parts) >= 2:
            return str(parts[-2])
        return str(parts[-1]) if parts else ""

    def _summarize_class_metrics(
        self,
        image_paths: List[str],
        y_true_labels: List[int],
        y_pred_labels: List[Optional[int]],
        label_map: List[str],
        ignore_invalid_predictions: bool = False,
    ) -> Dict[str, Any]:
        class_totals: Dict[str, int] = {str(lbl): 0 for lbl in label_map}
        class_correct: Dict[str, int] = {str(lbl): 0 for lbl in label_map}
        case_ids_by_class: Dict[str, List[str]] = {str(lbl): [] for lbl in label_map}
        case_ids_correct_by_class: Dict[str, List[str]] = {str(lbl): [] for lbl in label_map}

        for img_path, y_true, y_pred in zip(image_paths, y_true_labels, y_pred_labels):
            if y_true is None or int(y_true) < 0 or int(y_true) >= len(label_map):
                continue
            if ignore_invalid_predictions and (
                y_pred is None or int(y_pred) < 0 or int(y_pred) >= len(label_map)
            ):
                continue
            class_name = str(label_map[int(y_true)])
            class_totals[class_name] += 1
            case_id = self._extract_case_id_from_path(str(img_path), label_map)
            if case_id and case_id not in case_ids_by_class[class_name]:
                case_ids_by_class[class_name].append(case_id)
            if y_pred is not None and int(y_pred) == int(y_true):
                class_correct[class_name] += 1
                if case_id and case_id not in case_ids_correct_by_class[class_name]:
                    case_ids_correct_by_class[class_name].append(case_id)

        class_accuracy_pct: Dict[str, float] = {}
        for class_name in label_map:
            key = str(class_name)
            total = int(class_totals.get(key, 0))
            correct = int(class_correct.get(key, 0))
            class_accuracy_pct[key] = (100.0 * float(correct) / float(total)) if total > 0 else 0.0

        avg_class_accuracy_pct = (
            float(sum(class_accuracy_pct.values()) / len(class_accuracy_pct))
            if class_accuracy_pct
            else 0.0
        )

        return {
            "class_totals": class_totals,
            "class_correct": class_correct,
            "case_ids_by_class": case_ids_by_class,
            "case_ids_correct_by_class": case_ids_correct_by_class,
            "class_accuracy_pct": class_accuracy_pct,
            "avg_class_accuracy_pct": float(avg_class_accuracy_pct),
        }

    def _persist_fold_results(
        self,
        round_num: int,
        validation_metrics: Optional[Dict[str, Any]],
        active_set_metrics: Optional[Dict[str, Any]],
        **gen_kwargs,
    ) -> None:
        results_root_raw = str(gen_kwargs.get("results_root", "results"))
        results_root = (
            results_root_raw
            if os.path.isabs(results_root_raw)
            else os.path.join(os.path.dirname(__file__), results_root_raw)
        )
        dataset_token = _safe_path_token(gen_kwargs.get("dataset", "dataset"), default="dataset")
        fold_token = _safe_path_token(self.fold if self.fold is not None else gen_kwargs.get("fold"), default="na")
        selection_token = _safe_path_token(self.selection_artifact_token, default="mdl")
        fold_dir = os.path.join(results_root, dataset_token, f"fold-{fold_token}")
        os.makedirs(fold_dir, exist_ok=True)
        results_path = os.path.join(fold_dir, f"results_selection={selection_token}.json")

        payload: Dict[str, Any] = {}
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload.setdefault("dataset", str(gen_kwargs.get("dataset", "dataset")))
        payload.setdefault("fold", self.fold if self.fold is not None else gen_kwargs.get("fold"))
        payload.setdefault("selection_method", str(self.selection_method))
        payload.setdefault("created_at", now_ts)
        payload["updated_at"] = now_ts
        rounds = payload.setdefault("rounds", {})
        rounds[str(int(round_num))] = {
            "validation": validation_metrics or {},
            "active_set": active_set_metrics or {},
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _persist_final_test_results(
        self,
        test_metrics: Optional[Dict[str, Any]],
        **gen_kwargs,
    ) -> None:
        if test_metrics is None:
            return
        results_root_raw = str(gen_kwargs.get("results_root", "results"))
        results_root = (
            results_root_raw
            if os.path.isabs(results_root_raw)
            else os.path.join(os.path.dirname(__file__), results_root_raw)
        )
        dataset_token = _safe_path_token(gen_kwargs.get("dataset", "dataset"), default="dataset")
        fold_token = _safe_path_token(self.fold if self.fold is not None else gen_kwargs.get("fold"), default="na")
        selection_token = _safe_path_token(self.selection_artifact_token, default="mdl")
        fold_dir = os.path.join(results_root, dataset_token, f"fold-{fold_token}")
        os.makedirs(fold_dir, exist_ok=True)
        results_path = os.path.join(fold_dir, f"results_selection={selection_token}.json")

        payload: Dict[str, Any] = {}
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload.setdefault("dataset", str(gen_kwargs.get("dataset", "dataset")))
        payload.setdefault("fold", self.fold if self.fold is not None else gen_kwargs.get("fold"))
        payload.setdefault("selection_method", str(self.selection_method))
        payload.setdefault("created_at", now_ts)
        payload["updated_at"] = now_ts
        payload["test"] = test_metrics

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _prune_unlabeled_against_prompt_set(
        self,
        unlabeled_data: List[LabeledExample],
    ) -> int:
        """
        Remove unlabeled items whose image path already exists in the current prompt_set.
        Returns number of removed items.
        """
        if not unlabeled_data or not self.prompt_set:
            return 0
        prompt_paths = {os.path.abspath(str(x)) for x, _ in self.prompt_set}
        before = len(unlabeled_data)
        unlabeled_data[:] = [
            item for item in unlabeled_data
            if os.path.abspath(str(item[0])) not in prompt_paths
        ]
        return int(before - len(unlabeled_data))

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _to_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _sync_prompt_case_image_dirs(
        self,
        fold_dir: str,
        prompts: List[Dict[str, Any]],
        label_map: List[str],
    ) -> None:
        class_names = [str(lbl) for lbl in (label_map or [])]
        class_names_lower = {c.lower() for c in class_names}

        # Rebuild class/case image directories from the current global prompt list.
        for class_name in class_names:
            class_dir = os.path.join(fold_dir, class_name)
            if os.path.isdir(class_dir):
                shutil.rmtree(class_dir)

        for item in prompts:
            if not isinstance(item, dict):
                continue
            src_path = str(item.get("image_path", "")).strip()
            if not src_path:
                continue
            cls_raw = str(item.get("class", "")).strip()
            class_name = cls_raw if cls_raw.lower() in class_names_lower else ""
            if not class_name:
                inferred = self._infer_label_from_path(src_path, class_names)
                if inferred is None or inferred < 0 or inferred >= len(class_names):
                    continue
                class_name = class_names[inferred]
            case_id = self._extract_case_id_from_path(src_path, class_names)
            if not case_id:
                continue
            dst_case_dir = os.path.join(fold_dir, class_name, str(case_id))
            os.makedirs(dst_case_dir, exist_ok=True)
            if not os.path.exists(src_path):
                continue
            dst_path = os.path.join(dst_case_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

    def _rebuild_global_prompts_file(
        self,
        prompts_root: str,
        dataset_name: str,
        fold: Optional[int],
        selection_method: str,
        label_map: List[str],
    ) -> Optional[str]:
        dataset_token = _safe_path_token(dataset_name, default="dataset")
        fold_token = _safe_path_token(fold, default="na")
        selection_token = _safe_path_token((selection_method or self.selection_method), default="mdl")
        fold_dir = os.path.join(prompts_root, dataset_token, f"fold-{fold_token}_{selection_token}")
        os.makedirs(fold_dir, exist_ok=True)

        round_pattern = re.compile(
            rf"^.*round(\d+)_prompts_{re.escape(selection_token)}_corrected\.json$",
            flags=re.IGNORECASE,
        )
        round_files: List[Tuple[int, str]] = []
        for name in os.listdir(fold_dir):
            m = round_pattern.match(name)
            if not m:
                continue
            round_num = self._to_int(m.group(1), default=-1)
            if round_num < 0:
                continue
            round_files.append((round_num, os.path.join(fold_dir, name)))

        round_files.sort(key=lambda x: x[0])

        all_prompts: List[Dict[str, Any]] = [dict(x) for x in (self.initial_prompt_items or [])]
        total_duration = float(self._to_float(self.initial_global_metrics.get("Duration in minutes", 0.0), default=0.0))
        total_caption_corrections = int(self._to_int(self.initial_global_metrics.get("caption correction count", 0), default=0))
        round_accs: List[float] = []
        init_accuracy = self.initial_global_metrics.get("accuracy")

        for _, path in round_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            round_prompts = payload.get("prompts", [])
            if isinstance(round_prompts, list):
                all_prompts.extend(round_prompts)
            total_duration += self._to_float(payload.get("Duration in minutes", 0.0), default=0.0)
            total_caption_corrections += self._to_int(payload.get("caption correction count", 0), default=0)
            if "accuracy" in payload and payload.get("accuracy") not in (None, ""):
                try:
                    round_accs.append(float(payload.get("accuracy")))
                except (TypeError, ValueError):
                    pass

        global_payload = {
            "prompts": all_prompts,
            "caption correction count": int(total_caption_corrections),
            "Duration in minutes": float(total_duration),
            "accuracy": float(sum(round_accs) / len(round_accs)) if round_accs else init_accuracy,
        }
        global_path = os.path.join(fold_dir, f"global_prompts_{selection_token}_corrected.json")
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(global_payload, f, indent=4)

        try:
            self._sync_prompt_case_image_dirs(
                fold_dir=fold_dir,
                prompts=all_prompts,
                label_map=list(label_map or []),
            )
        except Exception as e:
            print(f"Warning: failed to sync class/case prompt image directories: {e}")

        return global_path

    def _evaluate_validation_subset(
        self,
        val_data: List[LabeledExample],
        round_num: int,
        label_map: Optional[List[str]],
        **gen_kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not val_data:
            return None
        if not label_map:
            print("Validation stopping check skipped: label_map is required.")
            return None

        by_class: Dict[int, List[str]] = {}
        for item in val_data:
            image_path = str(item[0])
            y_true = None
            if len(item) > 1 and isinstance(item[1], Integral):
                y_idx = int(item[1])
                if 0 <= y_idx < len(label_map):
                    y_true = y_idx
            # Fallback for malformed/legacy entries.
            if y_true is None:
                y_true = self._infer_label_from_path(image_path, label_map)
            if y_true is None:
                continue
            by_class.setdefault(int(y_true), []).append(image_path)

        if not by_class:
            print("Validation stopping check skipped: no class labels inferred from image paths.")
            return None

        eval_paths: List[str] = []
        eval_labels: List[int] = []
        sampled_per_class: Dict[str, int] = {}

        for class_idx in sorted(by_class.keys()):
            picked = list(by_class[class_idx])
            eval_paths.extend(picked)
            eval_labels.extend([class_idx] * len(picked))
            sampled_per_class[str(label_map[class_idx])] = int(len(picked))

        if not eval_paths:
            return None

        val_results_root = str(gen_kwargs.get("val_results_root", "val_results"))
        dataset_token = _safe_path_token(gen_kwargs.get("dataset", "dataset"), default="dataset")
        selection_token = _safe_path_token(self.selection_artifact_token, default="mdl")
        fold_token = _safe_path_token(self.fold if self.fold is not None else gen_kwargs.get("fold"), default="na")
        round_token = f"{int(round_num):02d}"
        val_dir = os.path.join(val_results_root, dataset_token, f"selection_method={selection_token}")
        os.makedirs(val_dir, exist_ok=True)
        val_predictions_path = os.path.join(val_dir, f"fold-{fold_token}.json")
        round_key = f"round_{round_token}"

        val_payload: Dict[str, Any] = {}
        if os.path.exists(val_predictions_path):
            try:
                with open(val_predictions_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    val_payload = existing
            except Exception:
                val_payload = {}

        val_payload.setdefault("dataset", str(gen_kwargs.get("dataset", "dataset")))
        val_payload.setdefault("fold", fold_token)
        val_payload.setdefault("selection_method", selection_token)
        val_payload.setdefault("split", "validation")
        val_payload.setdefault("round_predictions", {})

        def _build_val_records(
            current_preds: List[Optional[Tuple[Label, Caption]]],
        ) -> List[Dict[str, Any]]:
            records: List[Dict[str, Any]] = []
            for image_path, y_true, pred_pair in zip(eval_paths, eval_labels, current_preds):
                if pred_pair is None:
                    continue
                pred_raw = pred_pair[0] if isinstance(pred_pair, tuple) and len(pred_pair) > 0 else None
                rationale = pred_pair[1] if isinstance(pred_pair, tuple) and len(pred_pair) > 1 else ""
                y_true_idx = int(y_true)
                y_pred = self._normalize_prediction_label(pred_raw, label_map)
                y_pred_idx = int(y_pred) if y_pred is not None else None
                pred_label_name = (
                    str(label_map[y_pred_idx])
                    if y_pred_idx is not None and 0 <= y_pred_idx < len(label_map)
                    else "invalid"
                )
                records.append({
                    "round": int(round_num),
                    "image_path": str(image_path),
                    "ground_truth_label_idx": y_true_idx,
                    "ground_truth_label": str(label_map[y_true_idx]) if 0 <= y_true_idx < len(label_map) else None,
                    "pred_label_raw": pred_raw,
                    "pred_label_idx": y_pred_idx,
                    "pred_label": pred_label_name,
                    "caption": "" if rationale is None else str(rationale),
                    "is_correct": bool(y_pred_idx == y_true_idx) if y_pred_idx is not None else False,
                })
            return records

        def _persist_val_progress(records: List[Dict[str, Any]]) -> bool:
            try:
                val_payload["round_predictions"][round_key] = records
                with open(val_predictions_path, "w", encoding="utf-8") as f:
                    json.dump(val_payload, f, indent=2)
                return True
            except Exception as e:
                print(f"Warning: failed to write validation predictions file: {e}")
                return False

        query_batch_size = max(1, int(gen_kwargs.get("vlm_query_batch_size", 5)))
        eval_gen_kwargs = dict(gen_kwargs)
        # Ensure VLM parsing uses the same label mapping as validation ground truth.
        eval_gen_kwargs["label_map"] = list(label_map)

        preds: List[Optional[Tuple[Label, Caption]]] = [None] * len(eval_paths)
        resume_eval = bool(gen_kwargs.get("resume", False))
        resumed_count = 0
        if resume_eval:
            existing_records = val_payload.get("round_predictions", {}).get(round_key, [])
            records_by_path: Dict[str, Dict[str, Any]] = {}
            if isinstance(existing_records, list):
                for rec in existing_records:
                    if not isinstance(rec, dict):
                        continue
                    rec_path = rec.get("image_path")
                    if not rec_path:
                        continue
                    records_by_path[os.path.abspath(str(rec_path))] = rec
            for idx, image_path in enumerate(eval_paths):
                rec = records_by_path.get(os.path.abspath(str(image_path)))
                if not rec:
                    continue
                pred_raw = rec.get("pred_label_raw")
                if pred_raw is None:
                    pred_raw = rec.get("pred_label_idx")
                if pred_raw is None:
                    pred_raw = rec.get("pred_label")
                rationale = rec.get("caption", "")
                preds[idx] = (pred_raw, "" if rationale is None else str(rationale))
                resumed_count += 1
            if resumed_count > 0:
                print(
                    f"Validation r{int(round_num)}: resumed {resumed_count}/{len(eval_paths)} "
                    f"predictions from {val_predictions_path}."
                )

        pending_indices = [idx for idx, pred in enumerate(preds) if pred is None]
        pending_batches = (len(pending_indices) + query_batch_size - 1) // query_batch_size if pending_indices else 0
        for start in tqdm(
            range(0, len(pending_indices), query_batch_size),
            total=pending_batches,
            desc=f"Validation r{int(round_num)}",
            leave=False,
        ):
            batch_idx = (start // query_batch_size) + 1
            print(
                f"Validation r{int(round_num)} pending batch {batch_idx}/{pending_batches}..."
            )
            batch_indices = pending_indices[start:start + query_batch_size]
            batch_paths = [eval_paths[idx] for idx in batch_indices]
            batch_preds = vlm_query(
                batch_paths,
                self.S_1,
                self.S_2_template,
                self.prompt_set,
                stochastic=False,
                **eval_gen_kwargs,
            )
            if not isinstance(batch_preds, list):
                batch_preds = [batch_preds]
            if len(batch_preds) != len(batch_indices):
                print(
                    f"Validation stopping check skipped: batch prediction mismatch "
                    f"({len(batch_preds)} vs {len(batch_indices)})."
                )
                return None
            for idx, pred_pair in zip(batch_indices, batch_preds):
                preds[idx] = pred_pair

            _persist_val_progress(_build_val_records(preds))

        if any(pred is None for pred in preds):
            print(
                f"Validation stopping check skipped: prediction count mismatch "
                f"({sum(1 for p in preds if p is not None)} vs {len(eval_paths)})."
            )
            return None
        preds_final: List[Tuple[Label, Caption]] = [pred for pred in preds if pred is not None]
        pred_labels = [self._normalize_prediction_label(p[0], label_map) for p in preds_final]
        predicted_per_class: Dict[str, int] = {str(lbl): 0 for lbl in label_map}
        predicted_per_class["unknown"] = 0
        for y_pred in pred_labels:
            if y_pred is None or int(y_pred) < 0 or int(y_pred) >= len(label_map):
                predicted_per_class["unknown"] += 1
            else:
                predicted_per_class[str(label_map[int(y_pred)])] += 1

        class_metrics = self._summarize_class_metrics(
            image_paths=eval_paths,
            y_true_labels=[int(y) for y in eval_labels],
            y_pred_labels=[int(y) if y is not None else None for y in pred_labels],
            label_map=list(label_map),
            ignore_invalid_predictions=True,
        )

        final_records = _build_val_records([tuple(p) for p in preds_final])
        if not _persist_val_progress(final_records):
            val_predictions_path = ""

        return {
            "round": int(round_num),
            "validation_subset_size": int(len(eval_paths)),
            "sampled_per_class": sampled_per_class,
            "predicted_per_class": predicted_per_class,
            "class_totals": class_metrics["class_totals"],
            "class_correct": class_metrics["class_correct"],
            "case_ids_by_class": class_metrics["case_ids_by_class"],
            "case_ids_correct_by_class": class_metrics["case_ids_correct_by_class"],
            "class_accuracy_pct": class_metrics["class_accuracy_pct"],
            "avg_class_accuracy_pct": class_metrics["avg_class_accuracy_pct"],
            "val_predictions_path": val_predictions_path,
        }

    def _evaluate_test_subset(
        self,
        test_data: List[LabeledExample],
        round_num: int,
        label_map: Optional[List[str]],
        **gen_kwargs,
    ) -> Optional[Dict[str, Any]]:
        if not test_data:
            return None
        if not label_map:
            print("Test evaluation skipped: label_map is required.")
            return None

        by_class: Dict[int, List[str]] = {}
        for item in test_data:
            image_path = str(item[0])
            y_true = None
            if len(item) > 1 and isinstance(item[1], Integral):
                y_idx = int(item[1])
                if 0 <= y_idx < len(label_map):
                    y_true = y_idx
            if y_true is None:
                y_true = self._infer_label_from_path(image_path, label_map)
            if y_true is None:
                continue
            by_class.setdefault(int(y_true), []).append(image_path)

        if not by_class:
            print("Test evaluation skipped: no class labels inferred from image paths.")
            return None

        eval_paths: List[str] = []
        eval_labels: List[int] = []
        sampled_per_class: Dict[str, int] = {}
        for class_idx in sorted(by_class.keys()):
            picked = list(by_class[class_idx])
            eval_paths.extend(picked)
            eval_labels.extend([class_idx] * len(picked))
            sampled_per_class[str(label_map[class_idx])] = int(len(picked))

        if not eval_paths:
            return None

        test_results_root = str(gen_kwargs.get("test_results_root", "test_results"))
        dataset_token = _safe_path_token(gen_kwargs.get("dataset", "dataset"), default="dataset")
        selection_token = _safe_path_token(self.selection_artifact_token, default="mdl")
        fold_token = _safe_path_token(self.fold if self.fold is not None else gen_kwargs.get("fold"), default="na")
        round_token = f"{int(round_num):02d}"
        test_dir = os.path.join(test_results_root, dataset_token, f"selection_method={selection_token}")
        os.makedirs(test_dir, exist_ok=True)
        test_predictions_path = os.path.join(test_dir, f"fold-{fold_token}.json")
        round_key = f"round_{round_token}"

        test_payload: Dict[str, Any] = {}
        if os.path.exists(test_predictions_path):
            try:
                with open(test_predictions_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    test_payload = existing
            except Exception:
                test_payload = {}

        test_payload.setdefault("dataset", str(gen_kwargs.get("dataset", "dataset")))
        test_payload.setdefault("fold", fold_token)
        test_payload.setdefault("selection_method", selection_token)
        test_payload.setdefault("split", "test")
        test_payload.setdefault("round_predictions", {})

        def _build_test_records(
            current_preds: List[Optional[Tuple[Label, Caption]]],
        ) -> List[Dict[str, Any]]:
            records: List[Dict[str, Any]] = []
            for image_path, y_true, pred_pair in zip(eval_paths, eval_labels, current_preds):
                if pred_pair is None:
                    continue
                pred_raw = pred_pair[0] if isinstance(pred_pair, tuple) and len(pred_pair) > 0 else None
                rationale = pred_pair[1] if isinstance(pred_pair, tuple) and len(pred_pair) > 1 else ""
                y_true_idx = int(y_true)
                y_pred = self._normalize_prediction_label(pred_raw, label_map)
                y_pred_idx = int(y_pred) if y_pred is not None else None
                pred_label_name = (
                    str(label_map[y_pred_idx])
                    if y_pred_idx is not None and 0 <= y_pred_idx < len(label_map)
                    else "invalid"
                )
                records.append({
                    "round": int(round_num),
                    "image_path": str(image_path),
                    "ground_truth_label_idx": y_true_idx,
                    "ground_truth_label": str(label_map[y_true_idx]) if 0 <= y_true_idx < len(label_map) else None,
                    "pred_label_raw": pred_raw,
                    "pred_label_idx": y_pred_idx,
                    "pred_label": pred_label_name,
                    "caption": "" if rationale is None else str(rationale),
                    "is_correct": bool(y_pred_idx == y_true_idx) if y_pred_idx is not None else False,
                })
            return records

        def _persist_test_progress(records: List[Dict[str, Any]]) -> bool:
            try:
                test_payload["round_predictions"][round_key] = records
                # Keep backward compatibility with earlier readers.
                test_payload["round"] = int(round_num)
                test_payload["predictions"] = records
                with open(test_predictions_path, "w", encoding="utf-8") as f:
                    json.dump(test_payload, f, indent=2)
                return True
            except Exception as e:
                print(f"Warning: failed to write test predictions file: {e}")
                return False

        query_batch_size = max(1, int(gen_kwargs.get("vlm_query_batch_size", 5)))
        eval_gen_kwargs = dict(gen_kwargs)
        eval_gen_kwargs["label_map"] = list(label_map)

        preds: List[Optional[Tuple[Label, Caption]]] = [None] * len(eval_paths)
        resume_eval = bool(gen_kwargs.get("resume", False))
        resumed_count = 0
        if resume_eval:
            existing_records = test_payload.get("round_predictions", {}).get(round_key, [])
            if not existing_records and int(test_payload.get("round", -1)) == int(round_num):
                fallback_records = test_payload.get("predictions", [])
                if isinstance(fallback_records, list):
                    existing_records = fallback_records
            records_by_path: Dict[str, Dict[str, Any]] = {}
            if isinstance(existing_records, list):
                for rec in existing_records:
                    if not isinstance(rec, dict):
                        continue
                    rec_path = rec.get("image_path")
                    if not rec_path:
                        continue
                    records_by_path[os.path.abspath(str(rec_path))] = rec
            for idx, image_path in enumerate(eval_paths):
                rec = records_by_path.get(os.path.abspath(str(image_path)))
                if not rec:
                    continue
                pred_raw = rec.get("pred_label_raw")
                if pred_raw is None:
                    pred_raw = rec.get("pred_label_idx")
                if pred_raw is None:
                    pred_raw = rec.get("pred_label")
                rationale = rec.get("caption", "")
                preds[idx] = (pred_raw, "" if rationale is None else str(rationale))
                resumed_count += 1
            if resumed_count > 0:
                print(
                    f"Test r{int(round_num)}: resumed {resumed_count}/{len(eval_paths)} "
                    f"predictions from {test_predictions_path}."
                )

        pending_indices = [idx for idx, pred in enumerate(preds) if pred is None]
        pending_batches = (len(pending_indices) + query_batch_size - 1) // query_batch_size if pending_indices else 0
        for start in tqdm(
            range(0, len(pending_indices), query_batch_size),
            total=pending_batches,
            desc=f"Test r{int(round_num)}",
            leave=False,
        ):
            batch_idx = (start // query_batch_size) + 1
            print(f"Test r{int(round_num)} pending batch {batch_idx}/{pending_batches}...")
            batch_indices = pending_indices[start:start + query_batch_size]
            batch_paths = [eval_paths[idx] for idx in batch_indices]
            batch_preds = vlm_query(
                batch_paths,
                self.S_1,
                self.S_2_template,
                self.prompt_set,
                stochastic=False,
                **eval_gen_kwargs,
            )
            if not isinstance(batch_preds, list):
                batch_preds = [batch_preds]
            if len(batch_preds) != len(batch_indices):
                print(
                    f"Test evaluation skipped: batch prediction mismatch "
                    f"({len(batch_preds)} vs {len(batch_indices)})."
                )
                return None
            for idx, pred_pair in zip(batch_indices, batch_preds):
                preds[idx] = pred_pair

            _persist_test_progress(_build_test_records(preds))

        if any(pred is None for pred in preds):
            print(
                f"Test evaluation skipped: prediction count mismatch "
                f"({sum(1 for p in preds if p is not None)} vs {len(eval_paths)})."
            )
            return None

        preds_final: List[Tuple[Label, Caption]] = [pred for pred in preds if pred is not None]
        pred_labels = [self._normalize_prediction_label(p[0], label_map) for p in preds_final]
        predicted_per_class: Dict[str, int] = {str(lbl): 0 for lbl in label_map}
        predicted_per_class["unknown"] = 0
        for y_pred in pred_labels:
            if y_pred is None or int(y_pred) < 0 or int(y_pred) >= len(label_map):
                predicted_per_class["unknown"] += 1
            else:
                predicted_per_class[str(label_map[int(y_pred)])] += 1

        class_metrics = self._summarize_class_metrics(
            image_paths=eval_paths,
            y_true_labels=[int(y) for y in eval_labels],
            y_pred_labels=[int(y) if y is not None else None for y in pred_labels],
            label_map=list(label_map),
            ignore_invalid_predictions=True,
        )

        final_records = _build_test_records([tuple(p) for p in preds_final])
        if not _persist_test_progress(final_records):
            test_predictions_path = ""

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
            "test_predictions_path": test_predictions_path,
        }

    # -------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------
    def run(
        self,
        unlabeled_data: List[LabeledExample],
        val_data: List[LabeledExample],
        test_data: Optional[List[LabeledExample]] = None,
        **gen_kwargs
    ) -> PromptSet:
    
        initial_batch_size = gen_kwargs.get("initial_batch_size")
        resume = gen_kwargs.get("resume", False)
        checkpoint_path = with_selection_suffix(
            gen_kwargs.get("checkpoint_path", "checkpoint.json"),
            self.selection_artifact_token
        )

        # Keep a single mutable pool reference used by selection/removal/checkpointing.
        self.unlabeled_data = unlabeled_data
        
        start_round = 1
        
        if resume:
            if os.path.exists(checkpoint_path):
                try:
                    last_round, last_accuracy = self.load_checkpoint(checkpoint_path)
                    # On resume, checkpoint state is the source of truth for pool state.
                    unlabeled_data = self.unlabeled_data
                    start_round = last_round + 1
                    if last_accuracy is not None:
                        print(f"Last saved validation avg-class accuracy: {last_accuracy:.1f}%")
                    # If we finished max_rounds already
                    if start_round > self.max_rounds:
                        print(f"Checkpoint indicates {last_round} rounds completed. Max rounds is {self.max_rounds}.")
                        # Still run/resume final test evaluation before exiting.
                        test_eval_kwargs = dict(gen_kwargs)
                        test_eval_kwargs.pop("label_map", None)
                        test_metrics = self._evaluate_test_subset(
                            test_data=test_data or [],
                            round_num=int(last_round),
                            label_map=gen_kwargs.get("label_map"),
                            **test_eval_kwargs,
                        )
                        if test_metrics is None:
                            print("Final test evaluation skipped.")
                        else:
                            print(
                                f"Final test avg class accuracy (round {int(last_round)}): "
                                f"{float(test_metrics['avg_class_accuracy_pct']):.1f}% | "
                                f"classwise={test_metrics['class_accuracy_pct']} | "
                                f"sampled_per_class={test_metrics['sampled_per_class']} | "
                                f"predicted_per_class={test_metrics.get('predicted_per_class', {})} | "
                                f"pred_file={test_metrics.get('test_predictions_path', '')}"
                            )
                            try:
                                self._persist_final_test_results(
                                    test_metrics=test_metrics,
                                    **gen_kwargs,
                                )
                            except Exception as e:
                                print(f"Warning: failed to persist final test metrics: {e}")
                        return self.prompt_set
                except Exception as e:
                    print(f"Failed to resume from checkpoint: {e}. Starting from scratch.")
            else:
                print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

        removed_already_labeled = self._prune_unlabeled_against_prompt_set(unlabeled_data)
        self.unlabeled_data = unlabeled_data
        if removed_already_labeled > 0:
            print(
                f"Pruned {removed_already_labeled} items from unlabeled pool "
                "because they are already in prompt_set."
            )
        try:
            bootstrap_global = self._rebuild_global_prompts_file(
                prompts_root=str(gen_kwargs.get("prompts_root", "prompt_sets")),
                dataset_name=str(gen_kwargs.get("dataset", "dataset")),
                fold=self.fold if self.fold is not None else gen_kwargs.get("fold"),
                selection_method=self.selection_artifact_token,
                label_map=list(gen_kwargs.get("label_map") or []),
            )
            if bootstrap_global:
                print(f"Initialized/updated global prompts file: {bootstrap_global}")
        except Exception as e:
            print(f"Warning: failed to initialize global prompts from seed prompts: {e}")
        last_completed_round = max(0, int(start_round) - 1)

        dts_tuner = None
        dts_diagnostics_path = None
        if self.selection_method == "dts":
            from dts_diagnostics import DiagnosticsAndTuner

            dataset_name = str(gen_kwargs.get("dataset", "dataset"))
            fold_for_run = self.fold if self.fold is not None else gen_kwargs.get("fold")
            dts_run_outdir = build_dts_run_output_dir(
                base_outdir=self.diagnostic_outdir,
                dataset_name=dataset_name,
                fold=fold_for_run,
                selection_token=self.selection_artifact_token,
            )
            os.makedirs(dts_run_outdir, exist_ok=True)
            dts_diagnostics_path = os.path.join(dts_run_outdir, "diagnostics.jsonl")
            print(f"DTS diagnostics output directory: {dts_run_outdir}")
            dts_tuner = DiagnosticsAndTuner(
                diagnostics_jsonl_path=dts_diagnostics_path,
                diagnostic_outdir=dts_run_outdir,
                dataset_name=dataset_name,
                diagnostic_mode=self.diagnostic_mode,
                show_interactive=self.show_interactive,
                diagnostic_every=self.diagnostic_every,
                diagnostic_seed=self.diagnostic_seed,
                max_images_per_panel=self.max_images_per_panel,
                clip_model_name=self.dts_clip_model_name,
                clip_batch_size=self.clip_batch_size,
                use_mutual_knn=self.dts_mutual_knn,
            )
            if not self.dts_tune_hparams:
                print("DTS tuner is in log-only mode: hyperparameters will not be mutated.")

        for r in range(start_round, self.max_rounds + 1):
            print(f"\n=== Round {r} ===")


            # Precompute embeddings for P^(r-1)
            captions = [c for (_, c) in self.prompt_set]
            if len(captions) > 0:
                prompt_embeddings = text_encoder(captions)
            else:
                prompt_embeddings = np.zeros((0, 1), dtype=float)

            # Draw candidate subset U^(r)
            if len(unlabeled_data) == 0:
                print("Unlabeled pool exhausted. Stopping.")
                break

            # Optimized selection: Calculate scores for all candidates once
            # Iterative selection
            A_r = []
            selected_items = []
            
            # Intra-round checkpointing
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir, exist_ok=True)
                
            if self.fold is not None:
                intra_round_log_name = f"round_{r}_fold={self.fold}_selection={self.selection_artifact_token}_log.jsonl"
            else:
                intra_round_log_name = f"round_{r}_selection={self.selection_artifact_token}_log.jsonl"
            intra_round_log_path = os.path.join(self.logs_dir, intra_round_log_name)
            intra_round_embed_cache_path = intra_round_log_path.replace(
                "_log.jsonl",
                f"_{_safe_path_token(self.selection_artifact_token, default='dts')}_embeddings.npz",
            )

            # Backward-compatible resume: adopt known legacy intra-round naming patterns.
            if not os.path.exists(intra_round_log_path):
                legacy_names = [
                    f"round_{r}_log.jsonl",
                    f"round_{r}_log_{self.selection_method}.jsonl",
                    f"round_{r}_log.jsonl_{self.selection_method}.tmp",
                ]
                if self.fold is not None:
                    legacy_names.extend([
                        f"round_{r}_fold={self.fold}_log.jsonl",
                        f"round_{r}_fold={self.fold}_log_{self.selection_method}.jsonl",
                        f"fold={self.fold}_round_{r}_log.jsonl",
                        f"fold={self.fold}_round_{r}_log_{self.selection_method}.jsonl",
                    ])

                for legacy_name in legacy_names:
                    legacy_path = os.path.join(self.logs_dir, legacy_name)
                    if os.path.exists(legacy_path):
                        try:
                            os.rename(legacy_path, intra_round_log_path)
                            print(f"Migrated legacy intra-round log to: {intra_round_log_path}")
                        except Exception as e:
                            print(f"Warning: Could not migrate {legacy_path}: {e}")
                            intra_round_log_path = legacy_path
                        break
            
            # Load or Initialize Candidate List
            all_candidates = [] # List of {"score": ..., "item": ...}
            
            if os.path.exists(intra_round_log_path):
                print(f"Resuming round {r} from intra-round log: {intra_round_log_path}")
                try:
                    candidates_map = {} # image_path -> entry
                    with open(intra_round_log_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                item = tuple(entry["item"])
                                img_path = str(item[0])
                                candidates_map[img_path] = entry
                    
                    # Convert map values to list (preserving order is tricky if we use dict, 
                    # but python 3.7+ dicts preserve insertion order. 
                    # However, if we update an existing key, it stays in place.
                    # This is perfect: initial null entries set the order. 
                    # Later updates just change the value in place.)
                    all_candidates = list(candidates_map.values())
                    
                except Exception as e:
                    print(f"Error reading intra-round log: {e}. Starting fresh.")
                    all_candidates = []

            if not all_candidates:
                # Initialize new candidates
                if self.candidate_pool_size in (None, -1):
                    cand_size = len(unlabeled_data)
                else:
                    cand_size = min(self.candidate_pool_size, len(unlabeled_data))
                round_sampling_seed = int(self.diagnostic_seed) * 1000003 + int(r)
                U_r = random.Random(round_sampling_seed).sample(unlabeled_data, k=cand_size)
                all_candidates = [{"score": None, "item": item} for item in U_r]
                
                # Write initial state
                try:
                    with open(intra_round_log_path, 'w', encoding='utf-8') as f:
                        for entry in all_candidates:
                            f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    print(f"Error creating intra-round log: {e}")

            # Reconstruct U_r and candidate_scores from all_candidates
            U_r = [tuple(entry["item"]) for entry in all_candidates]
            candidate_scores = []
            dts_scores = None
            dts_meta = None
            dts_selected_indices = []
            dts_outlier_flags = np.array([], dtype=bool)
            dts_outlier_threshold = 0.0
            dts_selection_stats: Dict[str, Any] = {}
            round_sampling_seed: Optional[int] = None
            
            # Calculate scores for candidates that don't have them
            # We iterate through all_candidates and update in place
            
            # Populate existing scores
            processed_count = 0
            for entry in all_candidates:
                if entry["score"] is not None:
                    candidate_scores.append((entry["score"], tuple(entry["item"])))
                    processed_count += 1

            print(f"Already processed: {processed_count}/{len(all_candidates)}")
            selection_gen_kwargs = dict(gen_kwargs)
            selection_gen_kwargs["round_num"] = int(r)
            if selection_gen_kwargs.get("fold", None) is None and self.fold is not None:
                selection_gen_kwargs["fold"] = int(self.fold)

            if self.selection_method == "dts":
                # DTS scores are not independent per item, so rebuild candidate_scores from scratch.
                candidate_scores = []
                # For DTS, scores are computed jointly over the current candidate subset U_r.
                from dts_sampling import score_candidates_with_dts

                image_paths = [tuple(entry["item"])[0] for entry in all_candidates]
                dts_scores, dts_meta = score_candidates_with_dts(
                    image_paths=image_paths,
                    k=self.dts_k,
                    k_rho=self.dts_k_rho,
                    k_t=self.dts_k_t,
                    k_b=self.dts_k_b,
                    use_mutual_knn=self.dts_mutual_knn,
                    mcluster_min=self.dts_mcluster_min,
                    clip_model_name=self.dts_clip_model_name,
                    clip_batch_size=self.clip_batch_size,
                    embedding_cache_path=intra_round_embed_cache_path,
                )

                for idx, entry in enumerate(all_candidates):
                    entry["score"] = float(dts_scores[idx])
                    entry["dts_meta"] = {
                        "rho": float(dts_meta["rho"][idx]),
                        "root": int(dts_meta["root"][idx]),
                        "root_eff": int(dts_meta.get("root_eff", dts_meta["root"])[idx]),
                        "cluster_size": int(dts_meta["cluster_size"][idx]),
                        "cluster_size_eff": int(dts_meta.get("cluster_size_eff", dts_meta["cluster_size"])[idx]),
                        "mutual_degree": int(dts_meta["mutual_degree"][idx]),
                        "first_nn_distance": float(dts_meta["first_nn_distance"][idx]),
                    }

                # Rewrite entire log because DTS is a global computation.
                try:
                    with open(intra_round_log_path, 'w', encoding='utf-8') as f:
                        for entry in all_candidates:
                            f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    print(f"Error writing DTS intra-round log: {e}")

                for idx, entry in enumerate(all_candidates):
                    candidate_scores.append((float(dts_scores[idx]), tuple(entry["item"])))
            elif self.selection_method == "mdl":
                # Process remaining with MDL selection score
                with tqdm(total=len(all_candidates), initial=processed_count, desc="Calculating Scores") as pbar:
                    for i, entry in enumerate(all_candidates):
                        if entry["score"] is not None:
                            continue

                        item = tuple(entry["item"])
                        x, _ = item
                        s_x = selection_score(
                            x, self.S_1, self.S_2_template,
                            self.prompt_set,
                            prompt_embeddings,
                            lambda_c=self.lambda_c,
                            K=self.K_uncertainty,
                            alpha=self.alpha,
                            beta=self.beta,
                            **selection_gen_kwargs
                        )
                        # Update entry and list
                        entry["score"] = s_x
                        candidate_scores.append((s_x, item))

                        # Append to log file (Efficient O(1))
                        try:
                            with open(intra_round_log_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(entry) + "\n")
                        except Exception as e:
                            print(f"Error writing to intra-round log: {e}")

                        pbar.update(1)
            else:
                # Entropy-only selection score (APT-U Section 4).
                with tqdm(total=len(all_candidates), initial=processed_count, desc="Calculating Scores") as pbar:
                    for i, entry in enumerate(all_candidates):
                        if entry["score"] is not None:
                            continue

                        item = tuple(entry["item"])
                        x, _ = item
                        s_x = entropy_uncertainty(
                            x,
                            self.S_1,
                            self.S_2_template,
                            self.prompt_set,
                            K=self.K_uncertainty,
                            **selection_gen_kwargs,
                        )
                        entry["score"] = s_x
                        candidate_scores.append((s_x, item))

                        try:
                            with open(intra_round_log_path, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(entry) + "\n")
                        except Exception as e:
                            print(f"Error writing to intra-round log: {e}")

                        pbar.update(1)
            
            # Sort by score descending (stable); ties break by input order.
            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            
            num_to_select = min(initial_batch_size, len(candidate_scores))

            if self.selection_method == "dts":
                path_to_idx = {
                    str(tuple(entry["item"])[0]): idx
                    for idx, entry in enumerate(all_candidates)
                }
                idx_to_item = [tuple(entry["item"]) for entry in all_candidates]
                roots_raw = np.asarray(dts_meta["root"], dtype=np.int32)
                roots_eff = np.asarray(dts_meta.get("root_eff", dts_meta["root"]), dtype=np.int32)
                cluster_sizes_eff = np.asarray(dts_meta.get("cluster_size_eff", dts_meta["cluster_size"]), dtype=np.int32)
                first_nn_distance = np.asarray(dts_meta["first_nn_distance"], dtype=np.float32)
                deg_mut = np.asarray(dts_meta["mutual_degree"], dtype=np.int32)
                embeddings = np.asarray(dts_meta.get("embeddings")) if dts_meta.get("embeddings") is not None else None
                if first_nn_distance.size > 0:
                    dts_outlier_threshold = float(np.percentile(first_nn_distance, 99))
                    dts_outlier_flags = first_nn_distance > dts_outlier_threshold
                else:
                    dts_outlier_threshold = 0.0
                    dts_outlier_flags = np.array([], dtype=bool)

                ranked_candidates = []
                for score, item in candidate_scores:
                    idx = path_to_idx.get(str(item[0]))
                    if idx is None:
                        continue
                    ranked_candidates.append((idx, float(score), item))

                mcluster_min = int(self.dts_mcluster_min)
                base_cap = int(self.dts_max_per_basin)
                deg_min_tiny = int(self.dts_deg_min_tiny)
                b_min_tiny = float(self.dts_b_min_tiny)

                big_mask = cluster_sizes_eff >= mcluster_min
                tiny_mask = np.logical_not(big_mask)
                outlier_mask = (
                    np.asarray(dts_outlier_flags, dtype=bool)
                    if dts_outlier_flags.size == len(cluster_sizes_eff)
                    else np.zeros(len(cluster_sizes_eff), dtype=bool)
                )

                score_by_idx = {int(idx): float(score) for idx, score, _ in ranked_candidates}

                def _candidate_sort_key(idx: int) -> Tuple[float, int, int, float, int]:
                    # Deterministic ordering:
                    # 1) bscore desc, 2) basin size asc, 3) deg_mut desc, 4) d1(first-NN distance) desc, 5) index asc.
                    return (
                        -float(score_by_idx.get(int(idx), 0.0)),
                        int(cluster_sizes_eff[int(idx)]),
                        -int(deg_mut[int(idx)]),
                        -float(first_nn_distance[int(idx)]) if int(idx) < len(first_nn_distance) else 0.0,
                        int(idx),
                    )

                ranked_idx = sorted(
                    [int(idx) for idx, _, _ in ranked_candidates],
                    key=_candidate_sort_key,
                )

                selected_idx_set = set()
                basin_counts: Dict[int, int] = {}

                fill_strategy_counts = {
                    "relax_cap_big": 0,
                    "fallback_diversity_big": 0,
                    "fallback_tiny": 0,
                    "last_resort_outlier": 0,
                }
                rejection_reason_counts = {
                    "tiny_gated": 0,
                    "outlier_gated": 0,
                    "deg_mut_gated": 0,
                    "bscore_gated": 0,
                    "per_basin_cap": 0,
                }
                last_resort_outlier_counts = {"big": 0, "tiny": 0}

                def _add_selected(idx: int) -> None:
                    root_id = int(roots_eff[idx])
                    dts_selected_indices.append(int(idx))
                    selected_idx_set.add(int(idx))
                    selected_items.append(idx_to_item[idx])
                    A_r.append(idx_to_item[idx][0])
                    basin_counts[root_id] = int(basin_counts.get(root_id, 0)) + 1
                    print(f"Selected sample with score: {float(score_by_idx.get(int(idx), 0.0))}")

                def _try_select_idx(
                    idx: int,
                    *,
                    allow_tiny: bool,
                    allow_outliers: bool,
                    enforce_cap: bool,
                    cap_value: int,
                    require_tiny_thresholds: bool,
                ) -> bool:
                    if idx in selected_idx_set:
                        return False

                    is_tiny = bool(tiny_mask[idx])
                    is_outlier = bool(outlier_mask[idx]) if idx < outlier_mask.size else False
                    if is_tiny and not allow_tiny:
                        rejection_reason_counts["tiny_gated"] += 1
                        return False
                    if is_outlier and not allow_outliers:
                        rejection_reason_counts["outlier_gated"] += 1
                        return False
                    if is_tiny and require_tiny_thresholds:
                        if int(deg_mut[idx]) < deg_min_tiny:
                            rejection_reason_counts["deg_mut_gated"] += 1
                            return False
                        if float(score_by_idx.get(int(idx), 0.0)) < b_min_tiny:
                            rejection_reason_counts["bscore_gated"] += 1
                            return False
                    if enforce_cap:
                        root_id = int(roots_eff[idx])
                        current = int(basin_counts.get(root_id, 0))
                        if current >= int(cap_value):
                            rejection_reason_counts["per_basin_cap"] += 1
                            return False

                    _add_selected(int(idx))
                    return True

                def _pick_stage_round_robin(
                    candidate_pool: List[int],
                    *,
                    allow_tiny: bool,
                    allow_outliers: bool,
                    enforce_cap: bool,
                    cap_value: int,
                    require_tiny_thresholds: bool,
                    target_add: int,
                    skip_capped_basins_fast: bool = False,
                ) -> int:
                    if target_add <= 0:
                        return 0

                    pool = [int(idx) for idx in candidate_pool if int(idx) not in selected_idx_set]
                    if not pool:
                        return 0

                    # 1) Group by basin.
                    basin_to_candidates: Dict[int, List[int]] = {}
                    for idx in pool:
                        basin_id = int(roots_eff[idx])
                        basin_to_candidates.setdefault(basin_id, []).append(int(idx))

                    # 2) Sort candidates per basin by deterministic candidate key.
                    for basin_id, basin_candidates in basin_to_candidates.items():
                        basin_candidates.sort(key=_candidate_sort_key)
                        basin_to_candidates[basin_id] = basin_candidates

                    # 3) Sort basin order by best-candidate bscore desc, basin_size asc, basin_id asc.
                    basin_order = sorted(
                        basin_to_candidates.keys(),
                        key=lambda basin_id: (
                            -float(score_by_idx.get(int(basin_to_candidates[basin_id][0]), 0.0)),
                            int(cluster_sizes_eff[int(basin_to_candidates[basin_id][0])]),
                            int(basin_id),
                        ),
                    )

                    added = 0
                    active_basins = list(basin_order)
                    # 4) Round-robin over basins, picking top remaining from each basin.
                    while added < target_add and active_basins:
                        progressed = False
                        next_active_basins: List[int] = []
                        for basin_id in active_basins:
                            if added >= target_add:
                                break
                            if skip_capped_basins_fast and enforce_cap:
                                if int(basin_counts.get(int(basin_id), 0)) >= int(cap_value):
                                    continue
                            basin_candidates = basin_to_candidates.get(int(basin_id), [])
                            while basin_candidates:
                                if skip_capped_basins_fast and enforce_cap:
                                    if int(basin_counts.get(int(basin_id), 0)) >= int(cap_value):
                                        break
                                idx = int(basin_candidates.pop(0))
                                if _try_select_idx(
                                    idx,
                                    allow_tiny=allow_tiny,
                                    allow_outliers=allow_outliers,
                                    enforce_cap=enforce_cap,
                                    cap_value=cap_value,
                                    require_tiny_thresholds=require_tiny_thresholds,
                                ):
                                    added += 1
                                    progressed = True
                                    break
                            if basin_candidates:
                                if not (skip_capped_basins_fast and enforce_cap and int(basin_counts.get(int(basin_id), 0)) >= int(cap_value)):
                                    next_active_basins.append(int(basin_id))
                        active_basins = next_active_basins
                        if not progressed:
                            break
                    return int(added)

                def _format_candidate_row(idx: int) -> Dict[str, Any]:
                    return {
                        "index": int(idx),
                        "basin_id": int(roots_eff[idx]),
                        "basin_size": int(cluster_sizes_eff[idx]),
                        "bscore": float(score_by_idx.get(int(idx), 0.0)),
                        "deg_mut": int(deg_mut[idx]),
                        "d1": float(first_nn_distance[idx]) if idx < len(first_nn_distance) else None,
                        "outlier_flag": bool(outlier_mask[idx]) if idx < len(outlier_mask) else False,
                    }

                preview_idx = ranked_idx[:10]
                print(
                    "DTS top candidates (pre-selection): "
                    f"{json.dumps([_format_candidate_row(int(idx)) for idx in preview_idx])}"
                )

                def _build_stage_ranked_list(
                    candidate_pool: List[int],
                    *,
                    enforce_cap: bool,
                    cap_value: int,
                    require_tiny_thresholds: bool,
                    skip_capped_basins_fast: bool = False,
                    initial_selected: Optional[set] = None,
                    initial_basin_counts: Optional[Dict[int, int]] = None,
                ) -> Tuple[List[int], set, Dict[int, int]]:
                    local_selected = set(initial_selected or set())
                    local_basin_counts = dict(initial_basin_counts or {})
                    pool = [int(idx) for idx in candidate_pool if int(idx) not in local_selected]
                    if not pool:
                        return [], local_selected, local_basin_counts

                    basin_to_candidates: Dict[int, List[int]] = {}
                    for idx in pool:
                        basin_id = int(roots_eff[idx])
                        basin_to_candidates.setdefault(basin_id, []).append(int(idx))
                    for basin_id in basin_to_candidates.keys():
                        basin_to_candidates[basin_id].sort(key=_candidate_sort_key)

                    basin_order = sorted(
                        basin_to_candidates.keys(),
                        key=lambda basin_id: (
                            -float(score_by_idx.get(int(basin_to_candidates[basin_id][0]), 0.0)),
                            int(cluster_sizes_eff[int(basin_to_candidates[basin_id][0])]),
                            int(basin_id),
                        ),
                    )
                    active_basins = list(basin_order)
                    ranked: List[int] = []
                    while active_basins:
                        progressed = False
                        next_active_basins: List[int] = []
                        for basin_id in active_basins:
                            if skip_capped_basins_fast and enforce_cap:
                                if int(local_basin_counts.get(int(basin_id), 0)) >= int(cap_value):
                                    continue
                            basin_candidates = basin_to_candidates.get(int(basin_id), [])
                            while basin_candidates:
                                if skip_capped_basins_fast and enforce_cap:
                                    if int(local_basin_counts.get(int(basin_id), 0)) >= int(cap_value):
                                        break
                                idx = int(basin_candidates.pop(0))
                                if idx in local_selected:
                                    continue
                                if require_tiny_thresholds:
                                    if int(deg_mut[idx]) < deg_min_tiny:
                                        continue
                                    if float(score_by_idx.get(int(idx), 0.0)) < b_min_tiny:
                                        continue
                                if enforce_cap and int(local_basin_counts.get(int(basin_id), 0)) >= int(cap_value):
                                    continue
                                ranked.append(int(idx))
                                local_selected.add(int(idx))
                                local_basin_counts[int(basin_id)] = int(local_basin_counts.get(int(basin_id), 0)) + 1
                                progressed = True
                                break
                            if basin_candidates:
                                if not (skip_capped_basins_fast and enforce_cap and int(local_basin_counts.get(int(basin_id), 0)) >= int(cap_value)):
                                    next_active_basins.append(int(basin_id))
                        active_basins = next_active_basins
                        if not progressed:
                            break
                    return ranked, local_selected, local_basin_counts

                # Build deterministic ranked list per stage.
                DELTA_B = 0.10
                stage1_candidates = [int(idx) for idx in ranked_idx if bool(big_mask[idx]) and not bool(outlier_mask[idx])]
                stage1_bmax = float("nan")
                stage1_b_floor = float("nan")
                stage1_hi_pool: List[int] = []
                if stage1_candidates:
                    stage1_bmax = float(max(float(score_by_idx.get(int(idx), 0.0)) for idx in stage1_candidates))
                    stage1_b_floor = float(stage1_bmax - DELTA_B)
                    if bool(np.isfinite(stage1_bmax)):
                        stage1_hi_pool = [
                            int(idx)
                            for idx in stage1_candidates
                            if float(score_by_idx.get(int(idx), 0.0)) >= stage1_b_floor
                        ]
                stage1_hi_ranked, _, _ = _build_stage_ranked_list(
                    stage1_hi_pool,
                    enforce_cap=True,
                    cap_value=base_cap,
                    require_tiny_thresholds=False,
                    skip_capped_basins_fast=True,
                )
                stage1_rest_pool = [int(idx) for idx in stage1_candidates if int(idx) not in set(stage1_hi_ranked)]
                stage1_rest_ranked, _, _ = _build_stage_ranked_list(
                    stage1_rest_pool,
                    enforce_cap=True,
                    cap_value=base_cap,
                    require_tiny_thresholds=False,
                    skip_capped_basins_fast=True,
                )
                stage1_ranked = [int(idx) for idx in stage1_hi_ranked] + [int(idx) for idx in stage1_rest_ranked]
                stage1_hi_set = set(stage1_hi_ranked)

                relax_pool = [int(idx) for idx in ranked_idx if bool(big_mask[idx]) and not bool(outlier_mask[idx])]
                relax_ranked: List[int] = []
                relax_selected_local: set = set()
                relax_basin_counts_local: Dict[int, int] = {}
                cap_relaxed_to = int(base_cap)
                relaxed_cap = int(base_cap)
                while True:
                    relaxed_cap += 1
                    added_ranked, relax_selected_local, relax_basin_counts_local = _build_stage_ranked_list(
                        relax_pool,
                        enforce_cap=True,
                        cap_value=int(relaxed_cap),
                        require_tiny_thresholds=False,
                        skip_capped_basins_fast=True,
                        initial_selected=relax_selected_local,
                        initial_basin_counts=relax_basin_counts_local,
                    )
                    if not added_ranked:
                        break
                    relax_ranked.extend(int(idx) for idx in added_ranked)
                    cap_relaxed_to = int(relaxed_cap)

                diversity_pool = [int(idx) for idx in ranked_idx if bool(big_mask[idx]) and not bool(outlier_mask[idx])]
                diversity_ranked, _, _ = _build_stage_ranked_list(
                    diversity_pool,
                    enforce_cap=False,
                    cap_value=base_cap,
                    require_tiny_thresholds=False,
                    skip_capped_basins_fast=False,
                )

                tiny_pool = [int(idx) for idx in ranked_idx if bool(tiny_mask[idx]) and not bool(outlier_mask[idx])]
                tiny_ranked, _, _ = _build_stage_ranked_list(
                    tiny_pool,
                    enforce_cap=True,
                    cap_value=base_cap,
                    require_tiny_thresholds=True,
                    skip_capped_basins_fast=True,
                )

                outlier_big_pool = [int(idx) for idx in ranked_idx if bool(big_mask[idx]) and bool(outlier_mask[idx])]
                outlier_big_ranked, _, _ = _build_stage_ranked_list(
                    outlier_big_pool,
                    enforce_cap=False,
                    cap_value=base_cap,
                    require_tiny_thresholds=False,
                    skip_capped_basins_fast=False,
                )
                outlier_tiny_pool = [int(idx) for idx in ranked_idx if bool(tiny_mask[idx]) and bool(outlier_mask[idx])]
                outlier_tiny_ranked, _, _ = _build_stage_ranked_list(
                    outlier_tiny_pool,
                    enforce_cap=False,
                    cap_value=base_cap,
                    require_tiny_thresholds=False,
                    skip_capped_basins_fast=False,
                )
                outlier_ranked = [int(idx) for idx in outlier_big_ranked] + [int(idx) for idx in outlier_tiny_ranked]

                stage_specs: List[Dict[str, Any]] = [
                    {
                        "name": "stage1",
                        "priority": 0,
                        "order": 0,
                        "enforce_cap": True,
                        "cap_value": int(base_cap),
                        "pool_size": int(len(stage1_candidates)),
                        "ranked": [int(idx) for idx in stage1_ranked],
                    },
                    {
                        "name": "relax_cap_big",
                        "priority": 1,
                        "order": 1,
                        "enforce_cap": True,
                        "cap_value": int(cap_relaxed_to),
                        "pool_size": int(len(relax_pool)),
                        "ranked": [int(idx) for idx in relax_ranked],
                    },
                    {
                        "name": "fallback_diversity_big",
                        "priority": 2,
                        "order": 2,
                        "enforce_cap": False,
                        "cap_value": int(base_cap),
                        "pool_size": int(len(diversity_pool)),
                        "ranked": [int(idx) for idx in diversity_ranked],
                    },
                    {
                        "name": "fallback_tiny",
                        "priority": 3,
                        "order": 3,
                        "enforce_cap": True,
                        "cap_value": int(base_cap),
                        "pool_size": int(len(tiny_pool)),
                        "ranked": [int(idx) for idx in tiny_ranked],
                    },
                    {
                        "name": "last_resort_outlier",
                        "priority": 4,
                        "order": 4,
                        "enforce_cap": False,
                        "cap_value": int(base_cap),
                        "pool_size": int(len(outlier_big_pool) + len(outlier_tiny_pool)),
                        "ranked": [int(idx) for idx in outlier_ranked],
                    },
                ]

                for stage in stage_specs:
                    top5 = [_format_candidate_row(int(idx)) for idx in stage["ranked"][:5]]
                    print(
                        "DTS stage ranked list: "
                        f"{json.dumps({'stage_name': stage['name'], 'pool_size': int(stage['pool_size']), 'ranked_list_size': int(len(stage['ranked'])), 'top5': top5})}"
                    )

                # Global stage-aware selection loop with deterministic cross-stage priority.
                stage_cursor = {str(stage["name"]): 0 for stage in stage_specs}
                selected_count_by_stage = {str(stage["name"]): 0 for stage in stage_specs}
                per_basin_cap_rejections_by_stage = {str(stage["name"]): 0 for stage in stage_specs}
                per_basin_cap_rejections_stage1_hi = 0
                per_basin_cap_rejections_stage1_rest = 0
                global_selection_trace: List[Dict[str, Any]] = []

                while len(dts_selected_indices) < num_to_select:
                    stage_heads: List[Tuple[int, Dict[str, Any], int]] = []
                    for stage in stage_specs:
                        stage_name = str(stage["name"])
                        ranked_list = stage["ranked"]
                        ptr = int(stage_cursor[stage_name])
                        while ptr < len(ranked_list):
                            idx = int(ranked_list[ptr])
                            if idx in selected_idx_set:
                                ptr += 1
                                continue
                            basin_id = int(roots_eff[idx])
                            stage_enforce_cap = bool(stage.get("enforce_cap", True))
                            stage_cap_value = int(stage.get("cap_value", base_cap))
                            if stage_enforce_cap and int(basin_counts.get(basin_id, 0)) >= int(stage_cap_value):
                                rejection_reason_counts["per_basin_cap"] += 1
                                per_basin_cap_rejections_by_stage[stage_name] = int(per_basin_cap_rejections_by_stage.get(stage_name, 0)) + 1
                                if stage_name == "stage1":
                                    if int(idx) in stage1_hi_set:
                                        per_basin_cap_rejections_stage1_hi += 1
                                    else:
                                        per_basin_cap_rejections_stage1_rest += 1
                                ptr += 1
                                continue
                            break
                        stage_cursor[stage_name] = int(ptr)
                        if ptr < len(ranked_list):
                            stage_heads.append((int(idx), stage, int(ptr)))

                    if not stage_heads:
                        break

                    chosen_idx, chosen_stage, chosen_ptr = sorted(
                        stage_heads,
                        key=lambda head: (
                            -float(score_by_idx.get(int(head[0]), 0.0)),
                            int(head[1]["priority"]),
                            int(cluster_sizes_eff[int(head[0])]),
                            -int(deg_mut[int(head[0])]),
                            -float(first_nn_distance[int(head[0])]) if int(head[0]) < len(first_nn_distance) else 0.0,
                            int(head[0]),
                            int(head[1]["order"]),
                        ),
                    )[0]

                    _add_selected(int(chosen_idx))
                    chosen_stage_name = str(chosen_stage["name"])
                    selected_count_by_stage[chosen_stage_name] = int(selected_count_by_stage.get(chosen_stage_name, 0)) + 1
                    stage_cursor[chosen_stage_name] = int(chosen_ptr + 1)
                    global_selection_trace.append(
                        {
                            "stage_name": chosen_stage_name,
                            "index": int(chosen_idx),
                            "basin_id": int(roots_eff[int(chosen_idx)]),
                            "bscore": float(score_by_idx.get(int(chosen_idx), 0.0)),
                        }
                    )

                underfilled_before_fill = max(0, int(num_to_select - int(selected_count_by_stage.get("stage1", 0))))
                fill_strategy_counts["relax_cap_big"] = int(selected_count_by_stage.get("relax_cap_big", 0))
                fill_strategy_counts["fallback_diversity_big"] = int(selected_count_by_stage.get("fallback_diversity_big", 0))
                fill_strategy_counts["fallback_tiny"] = int(selected_count_by_stage.get("fallback_tiny", 0))
                fill_strategy_counts["last_resort_outlier"] = int(selected_count_by_stage.get("last_resort_outlier", 0))
                last_resort_outlier_counts["big"] = int(
                    sum(1 for idx in dts_selected_indices if bool(big_mask[int(idx)]) and bool(outlier_mask[int(idx)]))
                )
                last_resort_outlier_counts["tiny"] = int(
                    sum(1 for idx in dts_selected_indices if bool(tiny_mask[int(idx)]) and bool(outlier_mask[int(idx)]))
                )
                selected_from_hi_count = int(sum(1 for idx in dts_selected_indices if int(idx) in stage1_hi_set))
                selected_from_rest_count = int(selected_count_by_stage.get("stage1", 0) - selected_from_hi_count)
                stage1_scores = [float(score_by_idx.get(int(idx), 0.0)) for idx in dts_selected_indices if int(idx) in set(stage1_ranked)]
                stage1_sel_min = float(np.min(stage1_scores)) if stage1_scores else float("nan")
                stage1_sel_med = float(np.median(stage1_scores)) if stage1_scores else float("nan")
                stage1_sel_max = float(np.max(stage1_scores)) if stage1_scores else float("nan")
                print(
                    "DTS Stage1 hi-band stats: "
                    f"stage1_bmax={stage1_bmax:.6f}, "
                    f"stage1_b_floor={stage1_b_floor:.6f}, "
                    f"stage1_candidates_count={int(len(stage1_candidates))}, "
                    f"stage1_hi_count={int(len(stage1_hi_pool))}, "
                    f"selected_from_hi_count={int(selected_from_hi_count)}, "
                    f"selected_from_rest_count={int(selected_from_rest_count)}, "
                    f"per_basin_cap_rejections_stage1_hi={int(max(0, per_basin_cap_rejections_stage1_hi))}, "
                    f"per_basin_cap_rejections_stage1_rest={int(max(0, per_basin_cap_rejections_stage1_rest))}, "
                    f"selected_bscore_min={stage1_sel_min:.6f}, "
                    f"selected_bscore_median={stage1_sel_med:.6f}, "
                    f"selected_bscore_max={stage1_sel_max:.6f}"
                )
                print(f"DTS global selection trace: {json.dumps(global_selection_trace)}")
                print(f"DTS selected_count_by_stage: {json.dumps(selected_count_by_stage)}")
                print(f"DTS per_basin_cap_rejections_by_stage: {json.dumps(per_basin_cap_rejections_by_stage)}")

                if fill_strategy_counts["last_resort_outlier"] > 0:
                    print(
                        "LAST_RESORT_OUTLIER_FILL: "
                        f"big={int(last_resort_outlier_counts['big'])}, "
                        f"tiny={int(last_resort_outlier_counts['tiny'])}"
                    )

                if len(dts_selected_indices) != num_to_select:
                    available_counts = {
                        "total": int(len(ranked_idx)),
                        "big_non_outlier_not_selected": int(
                            sum(
                                1
                                for idx in ranked_idx
                                if idx not in selected_idx_set and bool(big_mask[idx]) and not bool(outlier_mask[idx])
                            )
                        ),
                        "eligible_tiny_non_outlier_not_selected": int(
                            sum(
                                1
                                for idx in ranked_idx
                                if idx not in selected_idx_set
                                and bool(tiny_mask[idx])
                                and not bool(outlier_mask[idx])
                                and int(deg_mut[idx]) >= deg_min_tiny
                                and float(score_by_idx.get(int(idx), 0.0)) >= b_min_tiny
                            )
                        ),
                        "outlier_not_selected": int(
                            sum(1 for idx in ranked_idx if idx not in selected_idx_set and bool(outlier_mask[idx]))
                        ),
                    }
                    stats_dump = {
                        "required_b": int(num_to_select),
                        "selected_count": int(len(dts_selected_indices)),
                        "underfilled_before_fill": int(underfilled_before_fill),
                        "fill_strategy_counts": fill_strategy_counts,
                        "rejection_reason_counts": rejection_reason_counts,
                        "available_counts": available_counts,
                        "mcluster_min": int(mcluster_min),
                        "max_per_basin": int(base_cap),
                        "deg_min_tiny": int(deg_min_tiny),
                        "b_min_tiny": float(b_min_tiny),
                    }
                    raise RuntimeError(f"DTS Fill-to-b failed: {json.dumps(stats_dump)}")

                selected_big_count = int(sum(bool(big_mask[idx]) for idx in dts_selected_indices))
                selected_tiny_count = int(sum(bool(tiny_mask[idx]) for idx in dts_selected_indices))
                selected_outlier_count = int(sum(bool(outlier_mask[idx]) for idx in dts_selected_indices))
                selected_roots_raw = [int(roots_raw[idx]) for idx in dts_selected_indices]
                selected_roots_eff = [int(roots_eff[idx]) for idx in dts_selected_indices]
                selected_rows = [_format_candidate_row(int(idx)) for idx in dts_selected_indices]
                selected_per_basin_counts = Counter(int(roots_eff[idx]) for idx in dts_selected_indices)
                print(f"DTS selected candidates (ordered): {json.dumps(selected_rows)}")
                print(
                    "DTS selected basin coverage: "
                    f"selected_unique_basins={int(len(selected_per_basin_counts))}, "
                    f"selected_per_basin_counts={json.dumps({str(k): int(v) for k, v in sorted(selected_per_basin_counts.items(), key=lambda kv: kv[0])})}"
                )
                dts_selection_stats = {
                    "selected_count": int(len(dts_selected_indices)),
                    "selected_big_count": int(selected_big_count),
                    "selected_tiny_count": int(selected_tiny_count),
                    "selected_outlier_count": int(selected_outlier_count),
                    "underfilled_before_fill": int(underfilled_before_fill),
                    "fill_strategy_counts": {k: int(v) for k, v in fill_strategy_counts.items()},
                    "rejection_reason_counts": {k: int(v) for k, v in rejection_reason_counts.items()},
                    "selected_in_tiny_basins": int(selected_tiny_count),
                    "deg_min_tiny": int(deg_min_tiny),
                    "b_min_tiny": float(b_min_tiny),
                    "selected_unique_basins_raw": int(len(set(selected_roots_raw))),
                    "selected_unique_basins_eff": int(len(set(selected_roots_eff))),
                    "selected_per_basin_counts_raw": {str(k): int(v) for k, v in Counter(selected_roots_raw).items()},
                    "selected_per_basin_counts_eff": {str(k): int(v) for k, v in Counter(selected_roots_eff).items()},
                    "round_sampling_seed": int(round_sampling_seed) if round_sampling_seed is not None else None,
                }

                print(
                    "DTS selection constraints: "
                    f"mcluster_min={self.dts_mcluster_min}, "
                    f"max_per_basin={self.dts_max_per_basin}, "
                    f"relaxed_per_basin_cap={cap_relaxed_to}, "
                    f"deg_min_tiny={self.dts_deg_min_tiny}, "
                    f"b_min_tiny={self.dts_b_min_tiny:.3f}, "
                    f"outlier_threshold={dts_outlier_threshold:.6f}"
                )
            else:
                # MDL/entropy: plain top-score selection.
                for i in range(num_to_select):
                    score, item = candidate_scores[i]
                    A_r.append(item[0])
                    selected_items.append(item)
                    print(f"Selected sample with score: {score}")
            
            print(f"Selected {len(A_r)} samples.")

            # Oracle editing to obtain A_e^(r)
            # Retrieve predictions from log file for the selected samples A_r
            predictions_for_oracle = None
            vlm_log_path = gen_kwargs.get("vlm_log_path")
            
            if gen_kwargs.get("debug", False) and vlm_log_path and os.path.exists(vlm_log_path):
                try:
                    with open(vlm_log_path, 'r') as f:
                        log_data = json.load(f)
                        
                    # Create a map for faster lookup
                    # Normalize paths to absolute for comparison
                    log_map = {os.path.abspath(entry["image_path"]): entry for entry in log_data}
                    predictions_for_oracle = []
                    
                    for img_path in A_r:
                        abs_path = os.path.abspath(str(img_path))
                        if abs_path in log_map:
                            entry = log_map[abs_path]
                            predictions_for_oracle.append((entry["label"], entry["rationale"]))
                        else:
                            # Fallback if not found
                            print(f"Warning: Image {img_path} not found in VLM log. Using default.")
                            predictions_for_oracle.append((0, "")) 
                except Exception as e:
                    print(f"Error reading VLM log: {e}")
                    predictions_for_oracle = None

            # Oracle editing to obtain A_e^(r)
            # Pass the list of images A_r directly to oracle with predictions
            self.last_active_set_metrics = None
            oracle_results = self.oracle_label_and_edit(
                A_r,
                predictions=predictions_for_oracle,
                round_num=r,
                **gen_kwargs
            )
            
            A_e_r: PromptSet = []
            selected_true_labels: List[Any] = []
            for x, (y_true, c_tilde) in zip(A_r, oracle_results):
                A_e_r.append((x, c_tilde))
                selected_true_labels.append(y_true)
            
            # Remove selected items from unlabeled_data
            for item in selected_items:
                try:
                    unlabeled_data.remove(item)
                except ValueError:
                    pass

            # Update prompt set
            self.prompt_set.extend(A_e_r)

            if self.selection_method == "dts" and dts_tuner is not None and dts_scores is not None and dts_meta is not None:
                current_hparams = {
                    "k": int(self.dts_k),
                    "k_rho": int(self.dts_k_rho),
                    "k_t": int(self.dts_k_t),
                    "k_b": int(self.dts_k_b),
                    "mcluster_min": int(self.dts_mcluster_min),
                    "c_tiny": int(self.dts_c_tiny),
                    "max_per_basin": int(self.dts_max_per_basin),
                    "deg_min_tiny": int(self.dts_deg_min_tiny),
                    "b_min_tiny": float(self.dts_b_min_tiny),
                }

                candidate_paths = [str(tuple(entry["item"])[0]) for entry in all_candidates]
                predicted_labels = self._load_candidate_predicted_labels(
                    candidate_paths=candidate_paths,
                    vlm_log_path=gen_kwargs.get("vlm_log_path"),
                    label_map=gen_kwargs.get("label_map"),
                )
                labeled_paths_so_far = [str(x) for x, _ in self.prompt_set]
                labeled_captions_so_far = [str(c) for _, c in self.prompt_set]
                # Purity diagnostics should use only the current-round candidate pool after selection,
                # not the full global unlabeled pool.
                selected_candidate_idx_set = set(int(i) for i in dts_selected_indices)
                candidate_paths_after_selection = [
                    str(path) for idx, path in enumerate(candidate_paths)
                    if int(idx) not in selected_candidate_idx_set
                ]

                # NOTE: diagnostics purity is intentionally computed after selection and oracle labels
                # are available, and never fed back into the current round selection decisions.
                try:
                    dts_selection_stats = dict(dts_selection_stats or {})
                    dts_selection_stats["dts_tune_hparams_enabled"] = bool(self.dts_tune_hparams)
                    diagnostics, next_hparams, next_tuner_state = dts_tuner.run_round_diagnostics(
                        round_index=r,
                        n_pool=len(all_candidates),
                        batch_size=num_to_select,
                        labeled_count=len(self.prompt_set),
                        unlabeled_count=len(unlabeled_data),
                        hyperparams=current_hparams,
                        roots=np.asarray(dts_meta["root"]),
                        roots_eff=np.asarray(dts_meta.get("root_eff", dts_meta["root"])),
                        basin_sizes=np.asarray(dts_meta["cluster_size"]),
                        basin_sizes_eff=np.asarray(dts_meta.get("cluster_size_eff", dts_meta["cluster_size"])),
                        boundary_scores=np.asarray(dts_scores),
                        selected_indices=[int(i) for i in dts_selected_indices],
                        selected_paths=[str(item[0]) for item in selected_items],
                        candidate_paths=candidate_paths,
                        mutual_mask=np.asarray(dts_meta["mutual_mask"]),
                        first_nn_distance=np.asarray(dts_meta["first_nn_distance"]),
                        state=self.dts_tuner_state,
                        embeddings=np.asarray(dts_meta.get("embeddings")) if dts_meta.get("embeddings") is not None else None,
                        knn_indices=np.asarray(dts_meta["knn_indices"]),
                        knn_distances=np.asarray(dts_meta["knn_distances"]),
                        rho=np.asarray(dts_meta["rho"]),
                        parent=np.asarray(dts_meta.get("parent")) if dts_meta.get("parent") is not None else None,
                        top1_nn_idx=np.asarray(dts_meta.get("top1_nn_idx")) if dts_meta.get("top1_nn_idx") is not None else None,
                        predicted_labels=predicted_labels,
                        selected_true_labels=selected_true_labels,
                        label_map=gen_kwargs.get("label_map"),
                        labeled_paths_so_far=labeled_paths_so_far,
                        labeled_captions_so_far=labeled_captions_so_far,
                        unlabeled_paths_after_selection=candidate_paths_after_selection,
                        selection_stats=dts_selection_stats,
                    )
                except Exception as e:
                    print(f"WARNING [DTS-DIAGNOSTICS]: non-blocking diagnostics failure: {e}")
                    diagnostics = {}
                    next_hparams = current_hparams
                    next_tuner_state = self.dts_tuner_state

                checks = diagnostics.get("health_checks", {})
                if checks.get("overmerged"):
                    print("WARNING [OVER-MERGED]: DTS basins appear over-collapsed.")
                if checks.get("fragmented"):
                    print("WARNING [FRAGMENTED]: DTS basins appear too fragmented.")
                if checks.get("mutual_sparse"):
                    print("WARNING [MUTUAL-SPARSE]: mutual-kNN graph is sparse.")
                if checks.get("outlier_heavy"):
                    print("WARNING [OUTLIER-HEAVY]: selected set has high outlier/tiny-basin concentration.")
                if checks.get("boundary_flat"):
                    print("WARNING [BOUNDARY-FLAT]: boundary scores are low-contrast.")
                if checks.get("diversity_low"):
                    print("WARNING [DIVERSITY-LOW]: selected set covers too few basins.")

                self.dts_tuner_state = next_tuner_state
                if self.dts_tune_hparams:
                    self.dts_k = int(next_hparams["k"])
                    self.dts_k_rho = int(next_hparams["k_rho"])
                    self.dts_k_t = int(next_hparams["k_t"])
                    self.dts_k_b = int(next_hparams["k_b"])
                    self.dts_mcluster_min = int(next_hparams["mcluster_min"])
                    self.dts_c_tiny = int(next_hparams["c_tiny"])
                    self.dts_max_per_basin = int(next_hparams["max_per_basin"])
                else:
                    print("DTS hyperparameter tuning is disabled; keeping current DTS hyperparameters unchanged.")

            stop_due_to_accuracy = False
            validation_avg_acc = None
            is_final_round = int(r) >= int(self.max_rounds)
            validation_metrics = None

            if is_final_round:
                print(
                    f"Skipping validation evaluation at final round {r} "
                    f"(max_rounds={self.max_rounds}); proceeding to final test after round completion."
                )
            else:
                eval_gen_kwargs = dict(gen_kwargs)
                # Avoid duplicate keyword when forwarding kwargs.
                eval_gen_kwargs.pop("label_map", None)
                validation_metrics = self._evaluate_validation_subset(
                    val_data=val_data,
                    round_num=r,
                    label_map=gen_kwargs.get("label_map"),
                    **eval_gen_kwargs,
                )
                if validation_metrics is None:
                    print("Validation accuracy unavailable for this round; skipping accuracy-based stop check.")
                else:
                    validation_avg_acc = float(validation_metrics["avg_class_accuracy_pct"])
                    self.last_validation_avg_class_accuracy = validation_avg_acc
                    print(
                        f"Validation avg class accuracy (round {r}): {validation_avg_acc:.1f}% | "
                        f"classwise={validation_metrics['class_accuracy_pct']} | "
                        f"sampled_per_class={validation_metrics['sampled_per_class']} | "
                        f"predicted_per_class={validation_metrics.get('predicted_per_class', {})} | "
                        f"pred_file={validation_metrics.get('val_predictions_path', '')}"
                    )
                    eval_log = dict(validation_metrics)
                    eval_log["selection_method"] = str(self.selection_method)
                    eval_log["fold"] = self.fold
                    eval_log["stopping_threshold_pct"] = float(self.stopping_accuracy)
                    eval_log["stop_due_to_validation"] = bool(validation_avg_acc >= self.stopping_accuracy)
                    try:
                        os.makedirs(self.logs_dir, exist_ok=True)
                        if self.fold is not None:
                            eval_log_name = f"validation_metrics_fold={self.fold}_selection={self.selection_artifact_token}.jsonl"
                        else:
                            eval_log_name = f"validation_metrics_selection={self.selection_artifact_token}.jsonl"
                        eval_log_path = os.path.join(self.logs_dir, eval_log_name)
                        with open(eval_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(eval_log) + "\n")
                    except Exception as e:
                        print(f"Warning: failed to append validation metrics log: {e}")

                    if validation_avg_acc >= self.stopping_accuracy:
                        print(
                            f"Stopping: validation avg class accuracy {validation_avg_acc:.1f}% "
                            f"reached threshold {self.stopping_accuracy:.1f}%."
                        )
                        stop_due_to_accuracy = True

            try:
                self._persist_fold_results(
                    round_num=r,
                    validation_metrics=validation_metrics,
                    active_set_metrics=self.last_active_set_metrics,
                    **gen_kwargs,
                )
            except Exception as e:
                print(f"Warning: failed to persist fold results summary: {e}")

            # Save checkpoint
            self.unlabeled_data = unlabeled_data
            checkpoint_accuracy = (
                validation_avg_acc
                if validation_avg_acc is not None
                else self.last_validation_avg_class_accuracy
            )
            self.save_checkpoint(r, checkpoint_accuracy, checkpoint_path)
            
            # Delete intra-round log after successful completion of the round
            if os.path.exists(intra_round_log_path):
                try:
                    os.remove(intra_round_log_path)
                    print(f"Deleted intra-round log: {intra_round_log_path}")
                except Exception as e:
                    print(f"Error deleting intra-round log: {e}")

            # Delete per-round CLIP embedding cache after successful completion.
            if os.path.exists(intra_round_embed_cache_path):
                try:
                    os.remove(intra_round_embed_cache_path)
                    print(f"Deleted embedding cache: {intra_round_embed_cache_path}")
                except Exception as e:
                    print(f"Error deleting embedding cache: {e}")

            if stop_due_to_accuracy:
                last_completed_round = int(r)
                break
            last_completed_round = int(r)

        test_eval_kwargs = dict(gen_kwargs)
        test_eval_kwargs.pop("label_map", None)
        test_metrics = self._evaluate_test_subset(
            test_data=test_data or [],
            round_num=last_completed_round,
            label_map=gen_kwargs.get("label_map"),
            **test_eval_kwargs,
        )
        if test_metrics is None:
            print("Final test evaluation skipped.")
        else:
            print(
                f"Final test avg class accuracy (round {last_completed_round}): "
                f"{float(test_metrics['avg_class_accuracy_pct']):.1f}% | "
                f"classwise={test_metrics['class_accuracy_pct']} | "
                f"sampled_per_class={test_metrics['sampled_per_class']} | "
                f"predicted_per_class={test_metrics.get('predicted_per_class', {})} | "
                f"pred_file={test_metrics.get('test_predictions_path', '')}"
            )
            try:
                self._persist_final_test_results(
                    test_metrics=test_metrics,
                    **gen_kwargs,
                )
            except Exception as e:
                print(f"Warning: failed to persist final test metrics: {e}")

        # Save prompt set
        if os.path.dirname(self.prompt_set_path):
            os.makedirs(os.path.dirname(self.prompt_set_path), exist_ok=True)

        with open(self.prompt_set_path, "w", encoding="utf-8") as f:
            # Ensure x is converted to string for consistency with checkpointing
            serializable_prompt_set = [(str(x), c) for x, c in self.prompt_set]
            json.dump(serializable_prompt_set, f, indent=4)

        return self.prompt_set

def parse_arguments():
    parser = argparse.ArgumentParser(description="APT-MDL: Minimum-Description-Length Active Prompt Tuning")

    parser.add_argument("--config", type=str, default="config.json", help="Path to JSON config file.")

    # Parse known args to check for config file
    config_defaults = {}
    args_temp, _ = parser.parse_known_args()
    if args_temp.config and os.path.exists(args_temp.config):
        try:
            with open(args_temp.config, 'r') as f:
                config_defaults = json.load(f)
            print(f"Loaded configuration from {args_temp.config}")
        except Exception as e:
            print(f"Error loading config file {args_temp.config}: {e}")

    # System prompt paths
    parser.add_argument("--system_prompt_1", type=str, default="system_prompt_1.md",
                        help="Path to the first system prompt file.")
    parser.add_argument("--system_prompt_2", type=str, default="system_prompt_2.md",
                        help="Path to the second system prompt file.")

    # Dataset and Prompts
    parser.add_argument("--init_prompts_path", type=str, default="init_prompts",
                        help="Path to initial prompts directory.")
    parser.add_argument("--dataset", type=str, required=False,
                        help="Name of the dataset.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold number (optional). If provided, sets default paths for data and init prompts.")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated folds to run in sequence (e.g., '5,6,10').")
    parser.add_argument("--val_json_path", type=str, default=None,
                        help="Path to validation json directory (optional if fold provided).")
    parser.add_argument("--test_json_path", type=str, default=None,
                        help="Path to test jsonl file (optional if fold provided).")
    parser.add_argument("--unlabeled_data_json_path", type=str, required=False,
                        help="Path to unlabeled data json file (optional if fold provided).")
    parser.add_argument("--vlm_log_path", type=str, default=None,
                        help="Path to JSON file for logging VLM responses (selection-method suffix added automatically).")
    parser.add_argument("--oracle_path", type=str, default="oracle.json",
                        help="Path to oracle cache JSON file.")
    parser.add_argument("--oracle_script_path", type=str, default=None,
                        help="Optional path to APT-v3/oracle.py. Defaults to ../APT-v3/oracle.py from this file.")
    parser.add_argument("--oracle_dataset_name", type=str, default=None,
                        help="Optional dataset name passed to oracle.py (e.g., Lurcher). Auto-resolved if omitted.")
    parser.add_argument("--prompts_root", type=str, default="prompt_sets",
                        help="Root directory where pre/post oracle round prompt JSONs are stored.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint if available.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.json",
                        help="Path to the checkpoint file (selection-method suffix added automatically).")
    parser.add_argument("--prompt_set_path", type=str, default="final_prompt_set.json",
                        help="Path to save the final prompt set (selection-method suffix added automatically).")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="Directory to save logs, including intra-round checkpoints.")
    parser.add_argument("--val_results_root", type=str, default="val_results",
                        help="Directory root for per-round validation prediction dumps.")
    parser.add_argument("--test_results_root", type=str, default="test_results",
                        help="Directory root for final test prediction dumps.")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Directory root for consolidated per-fold JSON summaries.")

    # Selection method and DTS parameters
    parser.add_argument("--selection_method", type=str, required=True, choices=["mdl", "entropy", "dts"],
                        help="Active-set scoring method: 'mdl', 'entropy' (Shannon entropy), or 'dts' (embedding density-tree).")
    parser.add_argument("--dts_k", type=int, default=60,
                        help="k for DTS neighborhood graph construction.")
    parser.add_argument("--dts_k_rho", type=int, default=30,
                        help="k_rho neighbors for DTS density proxy.")
    parser.add_argument("--dts_k_t", type=int, default=20,
                        help="k_t neighbor index used to define DTS local threshold radius.")
    parser.add_argument("--dts_k_b", type=int, default=15,
                        help="k_b neighbors used to compute DTS boundary score.")
    parser.add_argument("--dts_mutual_knn", action="store_true",
                        help="Use HYBRID mutual-kNN in DTS: rho and t_i from standard kNN; parent links and boundary use mutual-kNN.")
    parser.add_argument("--dts_mcluster_min", type=int, default=20,
                        help="Minimum cluster mass threshold for tiny-cluster safeguards in DTS.")
    parser.add_argument("--dts_c_tiny", type=int, default=1,
                        help="Maximum selections allowed per tiny cluster (size < dts_mcluster_min) in DTS.")
    parser.add_argument("--dts_max_per_basin", type=int, default=2,
                        help="Maximum selections allowed per basin in DTS (auto-tuner may set 1 or 2).")
    parser.add_argument("--dts_deg_min_tiny", type=int, default=10,
                        help="Tiny-basin fallback gate: minimum mutual degree for tiny basin candidates.")
    parser.add_argument("--dts_b_min_tiny", type=float, default=0.6,
                        help="Tiny-basin fallback gate: minimum boundary score for tiny basin candidates.")
    parser.add_argument("--dts_tune_hparams", dest="dts_tune_hparams", action="store_true",
                        help="Enable DTS hyperparameter tuning (default: enabled).")
    parser.add_argument("--no_dts_tune_hparams", dest="dts_tune_hparams", action="store_false",
                        help="Disable DTS hyperparameter mutation; diagnostics/tuner decisions are still logged.")
    parser.set_defaults(dts_tune_hparams=True)
    parser.add_argument("--dts_clip_model_alias", type=str, default=None,
                        choices=sorted(DTS_CLIP_MODEL_ALIASES.keys()),
                        help="DTS embedding model alias key (optional; if omitted, dts_clip_model_name is used, then default is biomedclip).")
    parser.add_argument("--dts_clip_model_name", type=str, default=None,
                        help="(Legacy fallback) DTS embedding model id or alias; ignored if dts_clip_model_alias is set.")
    parser.add_argument("--clip_batch_size", type=int, default=32,
                        help="Batch size used by DTS image embedding model.")
    parser.add_argument("--diagnostic_mode", action="store_true",
                        help="Enable DTS diagnostics artifacts (plots/panels/purity reports).")
    parser.add_argument("--show_interactive", action="store_true",
                        help="Show diagnostics figures interactively (diagnostic_mode must be enabled).")
    parser.add_argument("--diagnostic_every", type=int, default=2,
                        help="Save heavy diagnostics every N rounds when diagnostic_mode is enabled.")
    parser.add_argument("--diagnostic_outdir", type=str, default="diagnostics/",
                        help="Directory for DTS diagnostics JSONL and per-round artifacts.")
    parser.add_argument("--diagnostic_seed", type=int, default=0,
                        help="Random seed used by diagnostics downsampling/PCA helpers.")
    parser.add_argument("--max_images_per_panel", type=int, default=40,
                        help="Maximum images to render in any diagnostics panel.")

    # MDL parameters
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Coefficient for caption length in caption complexity.")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Coefficient for redundancy in caption complexity.")
    parser.add_argument("--lambda_mdl", type=float, default=0.1,
                        help="Lambda parameter for MDL loss (trade-off between error and DL).")
    parser.add_argument("--lambda_c", type=float, default=0.5,
                        help="Lambda parameter for selection score (trade-off between uncertainty and ECC).")
    parser.add_argument("--K_uncertainty", type=int, default=5,
                        help="Number of stochastic VLM calls for uncertainty estimation.")
    parser.add_argument("--mdl_tol", type=float, default=1e-3,
                        help="Tolerance for MDL loss convergence.")
    parser.add_argument("--stopping_accuracy", type=float, default=90.0,
                        help="Stop when validation average class accuracy (%) reaches this threshold.")
    parser.add_argument("--max_rounds", type=int, default=20,
                        help="Maximum number of active learning rounds.")
    parser.add_argument("--candidate_pool_size", type=int, default=None,
                        help="Size of the candidate pool sampled from the unlabeled data. "
                             "If omitted or set to -1, use the full unlabeled pool.")
    parser.add_argument("--initial_batch_size", type=int, default=10,
                        help="Number of examples to select in each active learning round.")
    parser.add_argument("--val_batch_size", type=int, default=5,
                        help="Batch size for VLM queries during validation.")
    parser.add_argument("--selection_batch_size", type=int, default=5,
                        help="Batch size for VLM queries during sample selection.")
    parser.add_argument("--vlm_query_batch_size", type=int, default=5,
                        help="Batch size for VLM calls inside oracle_label_and_edit.")
    parser.add_argument("--vlm_timeout_s", type=float, default=120.0,
                        help="Per-request timeout (seconds) for VLM API calls.")
    parser.add_argument("--uncertainty_cache_path", type=str, default="logs/uncertainty_cache.jsonl",
                        help="Path to shared JSONL cache for stochastic label counts used by MDL/entropy.")

    # VLM generation parameters (passed as gen_kwargs)
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="VLM model to use (e.g., gpt-4o).")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for VLM generation.")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum tokens for VLM generation.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter for VLM generation.")
    parser.add_argument("--label_map", nargs='*', type=str, default=None,
                        help="Optional list of strings to map VLM output labels (e.g., ['W', 'L']).")
    # Debug flag
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (use cached predictions and dummy oracle).")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    if args.dataset is None:
        parser.error("the following arguments are required: --dataset")
    if args.candidate_pool_size is not None and args.candidate_pool_size < -1:
        parser.error("--candidate_pool_size must be -1 (for full pool) or a non-negative integer.")
    if args.dts_mcluster_min < 0:
        parser.error("--dts_mcluster_min must be a non-negative integer.")
    if args.dts_c_tiny < 0:
        parser.error("--dts_c_tiny must be a non-negative integer.")
    if args.dts_max_per_basin not in (1, 2):
        parser.error("--dts_max_per_basin must be 1 or 2.")
    if args.dts_deg_min_tiny < 0:
        parser.error("--dts_deg_min_tiny must be a non-negative integer.")
    if args.dts_b_min_tiny < 0.0 or args.dts_b_min_tiny > 1.0:
        parser.error("--dts_b_min_tiny must be in [0, 1].")
    if args.diagnostic_every <= 0:
        parser.error("--diagnostic_every must be a positive integer.")
    if args.max_images_per_panel <= 0:
        parser.error("--max_images_per_panel must be a positive integer.")
    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.dataset == "microscopy_lurcher":
        args.label_map = ["wild", "lurcher"]

    fold_values: List[int] = []
    if args.folds:
        for token in str(args.folds).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                fold_values.append(int(token))
            except ValueError:
                raise ValueError(f"Invalid fold in --folds: '{token}'")
    if args.fold is not None:
        fold_values.append(int(args.fold))

    if not fold_values:
        raise ValueError("Provide at least one fold using --fold or --folds.")

    folds_to_run = sorted(set(fold_values))
    print(f"Running folds in order: {folds_to_run}")

    multi_fold_mode = len(folds_to_run) > 1

    def _fold_scoped_path(path_value: Optional[str], fold_num: int, default_name: str) -> str:
        raw = default_name if path_value is None else str(path_value)
        if "{fold}" in raw:
            return raw.replace("{fold}", str(fold_num))
        if raw == default_name:
            if default_name == "checkpoint.json":
                return f"checkpoint_fold={fold_num}.json"
            if default_name == "final_prompt_set.json":
                return f"final_prompt_set_fold={fold_num}.json"
            return raw
        if multi_fold_mode:
            p = Path(raw)
            if f"fold={fold_num}" not in p.stem and f"fold-{fold_num}" not in p.stem:
                return str(p.with_name(f"{p.stem}_fold={fold_num}{p.suffix}"))
        return raw

    for fold_num in folds_to_run:
        print(f"\n================ Fold {fold_num} ================")

        fold_seed = resolve_fold_random_seed(
            dataset_name=args.dataset,
            fold=fold_num,
            datasets_root="datasets",
        )
        fold_diagnostic_seed = int(args.diagnostic_seed)
        if fold_seed is not None:
            set_global_random_seed(fold_seed)
            fold_diagnostic_seed = int(fold_seed)
            print(f"Using fold random seed: {fold_seed} (dataset={args.dataset}, fold={fold_num})")
        else:
            print(
                f"Warning: could not resolve fold random seed for dataset={args.dataset}, fold={fold_num}. "
                "Run may be non-reproducible."
            )

        unlabeled_data_json_path = args.unlabeled_data_json_path
        val_json_path = args.val_json_path
        test_json_path = args.test_json_path
        if unlabeled_data_json_path is None:
            unlabeled_data_json_path = f"datasets/{args.dataset}/fold-{fold_num}/train.jsonl"
        if val_json_path is None:
            preferred_val_selected = f"datasets/{args.dataset}/fold-{fold_num}/val_selected.jsonl"
            fallback_val = f"datasets/{args.dataset}/fold-{fold_num}/val.jsonl"
            val_json_path = preferred_val_selected if os.path.exists(preferred_val_selected) else fallback_val
        if test_json_path is None:
            test_json_path = f"datasets/{args.dataset}/fold-{fold_num}/test.jsonl"

        checkpoint_path = _fold_scoped_path(args.checkpoint_path, fold_num, "checkpoint.json")
        prompt_set_path = _fold_scoped_path(args.prompt_set_path, fold_num, "final_prompt_set.json")
        vlm_log_path = args.vlm_log_path
        if vlm_log_path:
            vlm_log_path = _fold_scoped_path(vlm_log_path, fold_num, "vlm_log.json")

        resolved_dts_clip_model_name = _resolve_dts_clip_model_from_inputs(
            dts_clip_model_alias=args.dts_clip_model_alias,
            dts_clip_model_name=args.dts_clip_model_name,
        )
        selection_artifact_token = _selection_artifact_token(
            selection_method=args.selection_method,
            dts_clip_model_name=resolved_dts_clip_model_name,
            active_set_batch_size=args.initial_batch_size,
            candidate_pool_size=args.candidate_pool_size,
        )

        checkpoint_path = with_selection_suffix(checkpoint_path, selection_artifact_token)
        prompt_set_path = with_selection_suffix(prompt_set_path, selection_artifact_token)
        if vlm_log_path:
            vlm_log_path = with_selection_suffix(vlm_log_path, selection_artifact_token)

        aptmdl = APTMDL(
            system_prompt_1_path=args.system_prompt_1,
            system_prompt_2_path=args.system_prompt_2,
            selection_method=args.selection_method,
            alpha=args.alpha,
            beta=args.beta,
            lambda_mdl=args.lambda_mdl,
            lambda_c=args.lambda_c,
            K_uncertainty=args.K_uncertainty,
            mdl_tol=args.mdl_tol,
            max_rounds=args.max_rounds,
            candidate_pool_size=args.candidate_pool_size,
            dts_k=args.dts_k,
            dts_k_rho=args.dts_k_rho,
            dts_k_t=args.dts_k_t,
            dts_k_b=args.dts_k_b,
            dts_mutual_knn=args.dts_mutual_knn,
            dts_mcluster_min=args.dts_mcluster_min,
            dts_c_tiny=args.dts_c_tiny,
            dts_max_per_basin=args.dts_max_per_basin,
            dts_deg_min_tiny=args.dts_deg_min_tiny,
            dts_b_min_tiny=args.dts_b_min_tiny,
            dts_tune_hparams=args.dts_tune_hparams,
            dts_clip_model_name=resolved_dts_clip_model_name,
            diagnostic_mode=args.diagnostic_mode,
            show_interactive=args.show_interactive,
            diagnostic_every=args.diagnostic_every,
            diagnostic_outdir=args.diagnostic_outdir,
            diagnostic_seed=fold_diagnostic_seed,
            max_images_per_panel=args.max_images_per_panel,
            clip_batch_size=args.clip_batch_size,
            val_batch_size=args.val_batch_size,
            debug=args.debug,
            oracle_path=args.oracle_path,
            prompt_set_path=prompt_set_path,
            logs_dir=args.logs_dir,
            fold=fold_num,
            stopping_accuracy=args.stopping_accuracy,
            active_set_batch_size=args.initial_batch_size,
        )
        aptmdl.initialize_seed(args.init_prompts_path, args.dataset, fold=fold_num)
        aptmdl.unlabeled_data = load_data(unlabeled_data_json_path, args.label_map)
        print("Unlabeled data size: ", len(aptmdl.unlabeled_data))

        aptmdl.val_data = load_data(val_json_path, args.label_map)
        print("Validation data size: ", len(aptmdl.val_data))
        aptmdl.test_data = []
        if test_json_path and os.path.exists(test_json_path):
            aptmdl.test_data = load_data(test_json_path, args.label_map)
            print("Test data size: ", len(aptmdl.test_data))
        else:
            if test_json_path:
                print(f"Warning: test file not found at {test_json_path}. Final test evaluation will be skipped.")
            else:
                print("Warning: no test file path provided. Final test evaluation will be skipped.")

        aptmdl.run(
            unlabeled_data=aptmdl.unlabeled_data,
            val_data=aptmdl.val_data,
            test_data=aptmdl.test_data,
            initial_batch_size=args.initial_batch_size,
            selection_batch_size=args.selection_batch_size,
            vlm_query_batch_size=args.vlm_query_batch_size,
            vlm_timeout_s=args.vlm_timeout_s,
            uncertainty_cache_path=args.uncertainty_cache_path,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            label_map=args.label_map,
            vlm_log_path=vlm_log_path,
            debug=args.debug,
            resume=args.resume,
            checkpoint_path=checkpoint_path,
            prompt_set_path=prompt_set_path,
            dataset=args.dataset,
            logs_dir=args.logs_dir,
            val_results_root=args.val_results_root,
            test_results_root=args.test_results_root,
            results_root=args.results_root,
            oracle_script_path=args.oracle_script_path,
            oracle_dataset_name=args.oracle_dataset_name,
            prompts_root=args.prompts_root,
            fold=fold_num,
        )
