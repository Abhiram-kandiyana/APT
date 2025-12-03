"""
APT–MDL: Minimum-Description-Length Active Prompt Tuning
Dr. MAli overleaf
"""
import numpy as np
import random
from typing import List, Tuple, Any, Dict, Union
from collections import Counter
import base64
import os
import re
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import tiktoken
import argparse
import json
import subprocess
import tempfile
from utils import *

# Initialize Tokenizer (globally to avoid reloading)
# Using cl100k_base which is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
enc = tiktoken.get_encoding("cl100k_base")

# Initialize SentenceTransformer model (globally to avoid reloading)
# Using a lightweight, efficient model by default
st_model = SentenceTransformer('all-MiniLM-L6-v2')

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    temperature = gen_kwargs.get("temperature", 0.7)
    max_tokens = gen_kwargs.get("max_tokens", 1000)
    top_p = gen_kwargs.get("top_p", 1.0)

    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens,
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
            print(f"Error calling VLM: {e}")
            retry_count += 1
            time.sleep(2)
            
    # If retries exhausted, return default values
    print(f"Max retries ({max_retries}) exhausted. Defaulting to class 0 and default explanation.")
    default_label = 0
    default_rationale = "Cellular organization is compact with dense arrangement. The layering pattern is well-defined and normal. Purkinje cell layer appear prominent. The granule cell layer is thick with high cell density. Overall structure shows continuity, and the staining pattern has uniform intensity."
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
    c: Caption,
    prompt_set: PromptSet,
    prompt_embeddings: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.1,
) -> float:
    """
    C(c;P) = alpha * len(c) + beta * max_sim(psi(c), psi(c')).
    """
    tokens = tokenizer(c)
    L = len(tokens)
    length_penalty = alpha * L

    if len(prompt_set) == 0:
        redundancy_penalty = 0.0
    else:
        e_c = text_encoder([c])[0]
        e_c_norm = e_c / (np.linalg.norm(e_c) + 1e-8)

        P_norm = prompt_embeddings / (
            np.linalg.norm(prompt_embeddings, axis=1, keepdims=True) + 1e-8
        )
        sims = (P_norm @ e_c_norm)
        max_sim = float(np.max(sims))
        redundancy_penalty = beta * max_sim

    return length_penalty + redundancy_penalty


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

def uncertainty(
    x: Image,
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    K: int = 5,
    **gen_kwargs
) -> float:
    """
    u(x;P) = 1 - max_y p_hat(y | x), using K stochastic calls.
    """

    labels = []
    for i in range(K):
        cached_preds = {}
        if gen_kwargs.get("debug", False):
            cached_preds = load_cached_preds(gen_kwargs)
            
        if str(x) in cached_preds:
            y_hat = cached_preds[str(x)]['label']
        else:
            y_hat, _ = vlm_query(
                x, system_prompt_1, system_prompt_2, prompt_set,
                stochastic=True,
                **gen_kwargs
            )
        labels.append(int(y_hat))

    counts = Counter(labels)
    freq_max = max(counts.values())
    return 1.0 - freq_max / K


# ============================================================
# EXPECTED CAPTION COMPLEXITY
# ============================================================

def expected_caption_complexity(
    x: Image,
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    prompt_embeddings: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.1,
    **gen_kwargs
) -> float:
    """
    \\hat C(x;P) = C(e_hat(x;P);P).
    """
    cached_preds = {}
    if gen_kwargs.get("debug", False):
        cached_preds = load_cached_preds(gen_kwargs)

    if str(x) in cached_preds:
        caption_hat = f"{cached_preds[str(x)]['rationale']} C: {gen_kwargs.get('label_map')[cached_preds[str(x)]['label']]}"
    else:
        label, e_hat = vlm_query(
            x, system_prompt_1, system_prompt_2, prompt_set,
            stochastic=False,
            **gen_kwargs
        )
        caption_hat = f"{e_hat} C: {gen_kwargs.get('label_map')[label]}"

    return caption_complexity(
        caption_hat,
        prompt_set,
        prompt_embeddings,
        alpha=alpha,
        beta=beta,
    )


# ============================================================
# SELECTION SCORE
# ============================================================

def selection_score(
    x: Image,
    system_prompt_1: str,
    system_prompt_2: str,
    prompt_set: PromptSet,
    prompt_embeddings: np.ndarray,
    lambda_c: float = 0.5,
    K: int = 5,
    alpha: float = 0.01,
    beta: float = 0.1,
    **gen_kwargs
) -> float:
    """
    s(x;P) = u(x;P) - lambda_c * \\hat C(x;P).
    """
    u_x = uncertainty(x, system_prompt_1, system_prompt_2, prompt_set, K=K, **gen_kwargs)
    c_x = expected_caption_complexity(
        x, system_prompt_1, system_prompt_2, prompt_set, prompt_embeddings,
        alpha=alpha, beta=beta, **gen_kwargs
    )
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
        alpha: float = 0.01,
        beta: float = 0.1,
        lambda_mdl: float = 0.1,
        lambda_c: float = 0.5,
        K_uncertainty: int = 5,
        mdl_tol: float = 1e-3,
        max_rounds: int = 20,
        candidate_pool_size: int = 200,
        val_batch_size: int = 5,
        debug: bool = False,
        oracle_path: str = None,
        prompt_set_path: str = "final_prompt_set.json",
        logs_dir: str = "logs",
    ):
        self.S_1 = load_files(system_prompt_1_path)
        # System prompt 2 is now a template string handled in code, not loaded from file
        self.S_2_template = "Assume that you are forced to classify the {N} images below based on the treatment without any further information. Now use the information analyzed from above images to provide the rationale and classification for the below image. You should focus on six features for your classification: cellular organization, layering pattern, purkinje cells, granule cell layer, overall structure and staining pattern as described above. Your rationale should include descriptions of all six features. your response should be given in the exact format for each image. 'R:' should indicate the textual explanation of the image based on the features described above. 'C:' should indicate the classification based on your rationale - either 'lurcher' for Lurcher mutant group or 'wild' for wild-type/normal. Be concise and specific. Do not include anything else in the output."
        self.alpha = alpha
        self.beta = beta
        self.lambda_mdl = lambda_mdl
        self.lambda_c = lambda_c
        self.K_uncertainty = K_uncertainty
        self.mdl_tol = mdl_tol
        self.max_rounds = max_rounds
        self.candidate_pool_size = candidate_pool_size
        self.val_batch_size = val_batch_size
        self.debug = debug
        self.oracle_path = oracle_path
        self.prompt_set_path = prompt_set_path
        self.logs_dir = logs_dir

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
        pool_size = self.candidate_pool_size
        init_batch = gen_kwargs.get("initial_batch_size", 10)
        temp = gen_kwargs.get("temperature", 0.7)
        
        log_filename = f"{dataset_name}_K_{K}_rounds_{max_rounds}_pool_{pool_size}_batch_{init_batch}_temp_{temp}.log"
        log_path = os.path.join("logs", log_filename)
        
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
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
        Human oracle interface.
        
        Opens a GUI tool for the expert to review and correct labels and rationales.
        Returns (true_label, edited_caption) for each image.
        """
        is_batch = isinstance(x, list)
        images = x if is_batch else [x]

        # 1. Check Oracle Cache (oracle.json)
        cached_results = {}
        if self.oracle_path and os.path.exists(self.oracle_path):
            try:
                with open(self.oracle_path, 'r') as f:
                    oracle_data = json.load(f)
                # Create a map: absolute path -> {label, rationale}
                for entry in oracle_data:
                    abs_path = os.path.abspath(entry["image_path"])
                    cached_results[abs_path] = entry
            except Exception as e:
                print(f"Error reading oracle cache: {e}")

        # Identify which images need labeling
        images_to_label = []
        indices_to_label = []
        final_results_map = {} # Map index in 'images' to result
        
        for idx, img in enumerate(images):
            img_path = os.path.abspath(str(img))
            if img_path in cached_results:
                # Found in cache
                entry = cached_results[img_path]
                label_str = entry["class"] # Note: oracle.json uses "class", tool uses "label"
                rationale = entry["rationale"]
                
                # Map label string back to index
                label_map = gen_kwargs.get("label_map", ["wild", "lurcher"])
                label_idx = 0
                if label_str in label_map:
                    label_idx = label_map.index(label_str)
                
                caption = f"{rationale} C: {label_str}"
                final_results_map[idx] = (label_idx, caption)
                print(f"Found in oracle cache: {os.path.basename(img_path)}")
            else:
                images_to_label.append(img)
                indices_to_label.append(idx)

        # If all images were cached, return results
        if not images_to_label:
            results_list = [final_results_map[i] for i in range(len(images))]
            if is_batch:
                return results_list
            else:
                return results_list[0]

        # 2. Get initial predictions for remaining images
        # Filter predictions if provided, otherwise query VLM
        predictions_to_label = []
        if predictions is not None:
             # predictions corresponds to 'images' list. We need to pick those for 'images_to_label'
             for idx in indices_to_label:
                 predictions_to_label.append(predictions[idx])
        else:
            # Query VLM for initial guesses for images_to_label
            predictions_to_label = vlm_query(
                images_to_label, self.S_1, self.S_2_template, self.prompt_set,
                stochastic=False,
                **gen_kwargs
            )
            
        # 3. Prepare data for tool
        tool_data = []
        label_map = gen_kwargs.get("label_map", ["wild", "lurcher"]) # Default fallback
        
        for img, (pred_label_idx, pred_rationale) in zip(images_to_label, predictions_to_label):
            # Convert label index to string for UI
            if 0 <= pred_label_idx < len(label_map):
                label_str = label_map[pred_label_idx]
            else:
                label_str = label_map[0] # Fallback
                
            tool_data.append({
                "image_path": os.path.abspath(str(img)),
                "label": label_str,
                "rationale": pred_rationale
            })
            
        # 4. Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_in:
            json.dump(tool_data, tmp_in, indent=4)
            tmp_in_path = tmp_in.name
            
        tmp_out_path = tmp_in_path.replace('.json', '_out.json')
        
        # 5. Call the tool
        if self.debug:
            print(f"Debug mode: Skipping Oracle Tool UI for {len(images_to_label)} images. Auto-accepting predictions.")
            # Simulate tool output by just using the input data
            # We need to read tmp_in_path to get what we would have sent
            with open(tmp_in_path, 'r') as f:
                corrected_data = json.load(f)
            
            # Write to tmp_out_path so the rest of the code works as expected
            with open(tmp_out_path, 'w') as f:
                json.dump(corrected_data, f, indent=4)
        else:
            cmd = [
                "python",
                "apt_correction_tool_v2.py",
                "--input_json", tmp_in_path,
                "--output_json", tmp_out_path,
                "--label_map"
            ] + label_map
            
            print(f"Launching Oracle Tool for {len(images_to_label)} images...")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running oracle tool: {e}")
                raise e
            
        # 6. Read results
        if not os.path.exists(tmp_out_path):
            raise FileNotFoundError("Oracle tool did not save output file.")
            
        with open(tmp_out_path, 'r') as f:
            corrected_data = json.load(f)
            
        # 7. Update Oracle Cache and Final Results
        new_cache_entries = []
        
        for i, item in enumerate(corrected_data):
            label_str = item['label']
            rationale = item['rationale']
            img_path_abs = item['image_path']
            
            # Add to final results map
            original_idx = indices_to_label[i]
            
            # Map label string back to index
            label_idx = 0
            if label_str in label_map:
                label_idx = label_map.index(label_str)
            
            caption = f"{rationale} C: {label_str}"
            final_results_map[original_idx] = (label_idx, caption)
            
            # Prepare for cache update
            # Note: oracle.json uses "class" instead of "label"
            new_cache_entries.append({
                "rationale": rationale,
                "class": label_str,
                "image_path": img_path_abs # Storing absolute path to be safe, or relative if preferred? User said "image path".
                # To match existing format, let's see what's in oracle.json. 
                # It seems to have relative paths: "datasets/microscopy_lurcher/..."
                # But we are working with absolute paths in the code. 
                # Ideally we should store what we have. 
                # If we want to be consistent with existing file, we might need to relativize, 
                # but absolute is safer for matching. 
                # I will store the path as is (absolute) for now, as matching handles absolute.
            })
            
        # Append new entries to oracle.json
        if self.oracle_path:
            try:
                # Re-read to ensure we have latest (though we are single threaded here usually)
                current_oracle_data = []
                if os.path.exists(self.oracle_path):
                    with open(self.oracle_path, 'r') as f:
                        current_oracle_data = json.load(f)
                
                current_oracle_data.extend(new_cache_entries)
                
                oracle_dir = os.path.dirname(self.oracle_path)
                if oracle_dir and not os.path.exists(oracle_dir):
                    os.makedirs(oracle_dir, exist_ok=True)

                with open(self.oracle_path, 'w') as f:
                    json.dump(current_oracle_data, f, indent=4)
                print(f"Updated oracle cache with {len(new_cache_entries)} new entries.")
            except Exception as e:
                print(f"Error updating oracle cache: {e}")

        # Clean up temp files
        try:
            os.remove(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
        except:
            pass

        # Construct final list in order
        final_results = [final_results_map[i] for i in range(len(images))]

        if is_batch:
            return final_results
        else:
            return final_results[0]

    # -------------------------------------------------------
    # CHECKPOINTING
    # -------------------------------------------------------
    def save_checkpoint(self, round_num: int, mdl_loss_val: float, checkpoint_path: str):
        """
        Saves the current state to a JSON file.
        """
        # Convert prompt_set and unlabeled_data to serializable format
        # prompt_set is saved as a list of dictionaries: {"image_path": path, "caption": caption}
        # unlabeled_data is saved as a list of tuples: (image_path, label)
        
        state = {
            "round": round_num,
            "mdl_loss": mdl_loss_val,
            "prompt_set": [{"image_path": str(x), "caption": c} for x, c in self.prompt_set],
            "unlabeled_data": [(str(x), l) for x, l in self.unlabeled_data]
        }
        
        try:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)

            with open(checkpoint_path, 'w') as f:
                json.dump(state, f, indent=4)
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """
        Loads state from a checkpoint file.
        Returns (last_round, last_mdl_loss).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        try:
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
                
            # Load prompt_set from list of dictionaries
            self.prompt_set = [(item["image_path"], item["caption"]) for item in state["prompt_set"]]
            self.unlabeled_data = [(x, l) for x, l in state["unlabeled_data"]]
            
            print(f"Checkpoint loaded from {checkpoint_path}. Resuming from round {state['round'] + 1}")
            return state["round"], state["mdl_loss"]
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e

    # -------------------------------------------------------
    # INITIAL SEED
    # -------------------------------------------------------
    def initialize_seed(self, init_prompts_path: str, dataset: str):
        """
        Load seed examples from a JSON file.
        """

        file_path = os.path.join(init_prompts_path, dataset)
        
        # Check if file exists, if not try with .json extension
        if not os.path.exists(file_path) and os.path.exists(file_path + ".json"):
            file_path += ".json"
            
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Initial prompts file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)
            

        seed_examples = []
        if "prompt" in data:
            for item in data["prompt"]:
                rationale = item.get("rationale", "")
                label = item.get("class", "")
                # Format: rationale C:label
                caption = f"{rationale} C: {label}"
                seed_examples.append((item.get("image_path", ""), caption))
        
        self.prompt_set = list(seed_examples)

    # -------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------
    def run(
        self,
        unlabeled_data: List[LabeledExample],
        val_data: List[LabeledExample],
        **gen_kwargs
    ) -> PromptSet:
    
        # Initial MDL loss
        initial_batch_size = gen_kwargs.get("initial_batch_size")
        resume = gen_kwargs.get("resume", False)
        checkpoint_path = gen_kwargs.get("checkpoint_path", "checkpoint.json")
        
        start_round = 1
        L_prev = None
        
        if resume:
            if os.path.exists(checkpoint_path):
                try:
                    last_round, last_loss = self.load_checkpoint(checkpoint_path)
                    start_round = last_round + 1
                    L_prev = last_loss
                    # If we finished max_rounds already
                    if start_round > self.max_rounds:
                        print(f"Checkpoint indicates {last_round} rounds completed. Max rounds is {self.max_rounds}.")
                        return self.prompt_set
                except Exception as e:
                    print(f"Failed to resume from checkpoint: {e}. Starting from scratch.")
            else:
                print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

        if L_prev is None:
            # Initial MDL loss
            print("Performing initial evaluation...")
            L_prev = self.evaluate(val_data, round_num=0, **gen_kwargs)
            print(f"Initial MDL loss: {L_prev:.4f}")

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
                
            intra_round_log_path = os.path.join(self.logs_dir, f"round_{r}_log.jsonl")
            
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
                cand_size = min(self.candidate_pool_size, len(unlabeled_data))
                U_r = random.sample(unlabeled_data, k=cand_size)
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
            
            # Calculate scores for candidates that don't have them
            # We iterate through all_candidates and update in place
            
            # Populate existing scores
            processed_count = 0
            for entry in all_candidates:
                if entry["score"] is not None:
                    candidate_scores.append((entry["score"], tuple(entry["item"])))
                    processed_count += 1

            print(f"Already processed: {processed_count}/{len(all_candidates)}")
            # Process remaining
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
                        **gen_kwargs
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
            
            
            # Sort by score descending
            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Select top initial_batch_size
            num_to_select = min(initial_batch_size, len(candidate_scores))
            for i in range(num_to_select):
                score, item = candidate_scores[i]
                A_r.append(item[0])
                selected_items.append(item)
                print(f"Selected sample with score: {score}")
            
            print(f"Selected {len(A_r)} samples.")

            # Oracle editing to obtain A_e^(r)
            # Retrieve predictions from log file for the selected samples A_r
            predictions_for_oracle = []
            vlm_log_path = gen_kwargs.get("vlm_log_path")
            
            if vlm_log_path and os.path.exists(vlm_log_path):
                try:
                    with open(vlm_log_path, 'r') as f:
                        log_data = json.load(f)
                        
                    # Create a map for faster lookup
                    # Normalize paths to absolute for comparison
                    log_map = {os.path.abspath(entry["image_path"]): entry for entry in log_data}
                    
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
            else:
                predictions_for_oracle = None

            # Oracle editing to obtain A_e^(r)
            # Pass the list of images A_r directly to oracle with predictions
            oracle_results = self.oracle_label_and_edit(A_r, predictions=predictions_for_oracle, **gen_kwargs)
            
            A_e_r: PromptSet = []
            for x, (y_true, c_tilde) in zip(A_r, oracle_results):
                A_e_r.append((x, c_tilde))
            
            # Remove selected items from unlabeled_data
            for item in selected_items:
                try:
                    unlabeled_data.remove(item)
                except ValueError:
                    pass

            # Update prompt set
            self.prompt_set.extend(A_e_r)

            # Compute MDL loss and stopping condition
            # Compute MDL loss and stopping condition
            L_curr = self.evaluate(val_data, round_num=r, **gen_kwargs)
            print(f"MDL loss: prev={L_prev:.4f}, curr={L_curr:.4f}")

            if abs(L_curr - L_prev) < self.mdl_tol:
                print("MDL loss converged. Stopping.")
                break

            L_prev = L_curr
            
            # Save checkpoint
            self.save_checkpoint(r, L_curr, checkpoint_path)
            
            # Delete intra-round log after successful completion of the round
            if os.path.exists(intra_round_log_path):
                try:
                    os.remove(intra_round_log_path)
                    print(f"Deleted intra-round log: {intra_round_log_path}")
                except Exception as e:
                    print(f"Error deleting intra-round log: {e}")

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

    # System prompt paths
    parser.add_argument("--system_prompt_1", type=str, default="system_prompt_1.md",
                        help="Path to the first system prompt file.")
    parser.add_argument("--system_prompt_2", type=str, default="system_prompt_2.md",
                        help="Path to the second system prompt file.")

    # Dataset and Prompts
    parser.add_argument("--init_prompts_path", type=str, default="init_prompts",
                        help="Path to initial prompts directory.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of the dataset.")
    parser.add_argument("--val_json_path", type=str, default="validation.json",
                        help="Path to validation json directory.")
    parser.add_argument("--unlabeled_data_json_path", type=str, required=True,
                        help="Path to unlabeled data json file.")
    parser.add_argument("--vlm_log_path", type=str, default=None,
                        help="Path to JSON file for logging VLM responses.")
    parser.add_argument("--oracle_path", type=str, default="oracle.json",
                        help="Path to oracle cache JSON file.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint if available.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.json",
                        help="Path to the checkpoint file.")
    parser.add_argument("--prompt_set_path", type=str, default="final_prompt_set.json",
                        help="Path to save the final prompt set.")
    parser.add_argument("--logs_dir", type=str, default="logs",
                        help="Directory to save logs, including intra-round checkpoints.")

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
    parser.add_argument("--max_rounds", type=int, default=20,
                        help="Maximum number of active learning rounds.")
    parser.add_argument("--candidate_pool_size", type=int, default=200,
                        help="Size of the candidate pool sampled from the unlabeled data.")
    parser.add_argument("--initial_batch_size", type=int, default=10,
                        help="Number of examples to select in each active learning round.")
    parser.add_argument("--val_batch_size", type=int, default=5,
                        help="Batch size for VLM queries during validation.")

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

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.dataset == "microscopy_lurcher":
        args.label_map = ["wild", "lurcher"]
    
    # Convert args to kwargs for APTMDL init
    # Note: APTMDL init expects system_prompt_1_path, but args has system_prompt_1
    aptmdl = APTMDL(
        system_prompt_1_path=args.system_prompt_1,
        system_prompt_2_path=args.system_prompt_2,
        alpha=args.alpha,
        beta=args.beta,
        lambda_mdl=args.lambda_mdl,
        lambda_c=args.lambda_c,
        K_uncertainty=args.K_uncertainty,
        mdl_tol=args.mdl_tol,
        max_rounds=args.max_rounds,

        candidate_pool_size=args.candidate_pool_size,
        val_batch_size=args.val_batch_size,
        debug=args.debug,
        oracle_path=args.oracle_path,
        prompt_set_path=args.prompt_set_path,
        logs_dir=args.logs_dir
    )
    # Initialize seed prompts
    aptmdl.initialize_seed(args.init_prompts_path, args.dataset)
    # Load unlabeled data initially (will be overwritten if resuming)
    aptmdl.unlabeled_data = load_data(args.unlabeled_data_json_path, args.label_map)
    print("Unlabeled data size: ", len(aptmdl.unlabeled_data))

    aptmdl.val_data = load_data(args.val_json_path, args.label_map)
    print("Validation data size: ", len(aptmdl.val_data))

    aptmdl.run(
        unlabeled_data=aptmdl.unlabeled_data,
        val_data=aptmdl.val_data,
        initial_batch_size=args.initial_batch_size,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        label_map=args.label_map,
        vlm_log_path=args.vlm_log_path,
        debug=args.debug,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        prompt_set_path=args.prompt_set_path,
        dataset=args.dataset,
        logs_dir=args.logs_dir
    )

