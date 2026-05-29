import json
import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path

try:
    from estimate_annotation_time import estimate_annotation_time
except Exception:
    # Fallback if optional plotting dependencies used by estimate_annotation_time are missing.
    def estimate_annotation_time(num_correct, num_total, t_fast=0.473, t_slow=0.840):
        if num_total <= 0:
            return 0.0
        p_correct = max(0.0, min(1.0, num_correct / num_total))
        return (p_correct * t_fast) + ((1.0 - p_correct) * t_slow)

def load_json(file_path):
    """Loads a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        sys.exit(1)

def save_json(data, file_path):
    """Saves data to a JSON file."""
    try:
        out_dir = os.path.dirname(file_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved output to {file_path}")
    except Exception as e:
        print(f"Error saving file to {file_path}: {e}")
        sys.exit(1)

def get_ground_truth(image_path, class_list):
    """
    Determines the ground truth class from the image path.
    Searches for the first directory in the path that matches a class in class_list (case-insensitive).
    """
    parts = image_path.split(os.sep)
    # Handle cases where path might use forward slashes on non-Unix systems or vice-versa if inconsistent
    if len(parts) == 1:
        parts = image_path.split('/')
        
    class_list_lower = [c.lower() for c in class_list]
    
    for part in parts:
        if part.lower() in class_list_lower:
            # Return the class name with the casing found in the class list (Title Case usually preferred for output)
            # Find the original casing from the list
            index = class_list_lower.index(part.lower())
            return class_list[index]
            
    return None

def parse_image_name(image_path):
    """
    Parses the image name to extract Case ID, Slide Number, Section Number, and Image Index.
    
    Logic:
    - Split filename (without extension) by underscores.
    - Right-to-Left:
        - Last part: Image Index
        - 2nd from right: 'Image' keyword (ignored)
        - 3rd from right: Section Number
        - 4th from right: Slide Number
    - Left-to-Right:
        - First part: Case ID
        
    Returns:
        dict: {
            "case_id": str,
            "slide_num": str,
            "section_num": str,
            "image_idx": str
        }
    or None if parsing fails.
    """
    try:
        basename = os.path.basename(image_path)
        root, _ = os.path.splitext(basename)
        parts = root.split('_')
        
        if len(parts) < 5: # Need at least Case, ..., Slide, Section, Image, Index
            return None
            
        # Extract from right
        image_idx = parts[-1]
        # parts[-2] should be 'Image'
        section_num = parts[-3]
        slide_num = parts[-4]
        
        # Extract from left
        case_id = parts[0]
        
        return {
            "case_id": case_id,
            "slide_num": slide_num,
            "section_num": section_num,
            "image_idx": image_idx
        }
    except Exception as e:
        print(f"Error parsing image name {image_path}: {e}")
        return None


def resolve_prompt_bank_path(dataset_name: str, apt_v3_dir: Path) -> Path:
    """
    Resolve prompt bank file path from a possibly project-specific dataset name.
    """
    ds = (dataset_name or "").strip()
    ds_lower = ds.lower()
    repo_root = apt_v3_dir.parents[1]

    candidate_names = [
        "microscopy_lurcher.json",
        "oracle.json",
        "prompt_bank.json",
        f"{ds}Data_prompt_bank.json",
        f"{ds}_prompt_bank.json",
    ]
    if "lurcher" in ds_lower:
        candidate_names.extend(["LurcherData_prompt_bank.json", "lurcherData_prompt_bank.json"])
    if "fengru" in ds_lower:
        candidate_names.extend(["FengRuData_prompt_bank.json", "fengruData_prompt_bank.json"])

    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        for p in [
            apt_v3_dir / name,
            repo_root / "prompt_banks" / name,
            repo_root / "prompt_banks" / "microscopy_lurcher" / name,
        ]:
            if p.exists():
                return p
    # Fallback: pick first prompt bank with dataset token in filename.
    for p in sorted(apt_v3_dir.glob("*prompt_bank.json")):
        if ds_lower and ds_lower in p.stem.lower():
            return p

    raise FileNotFoundError(
        f"Could not find prompt bank for dataset '{dataset_name}'. "
        "Pass --prompt_bank_path or download prompt_banks/microscopy_lurcher.json."
    )

def run_manual_corrections(manual_items, class_list, correction_tool_path):
    """
    Run batch manual correction tool for unresolved wrong predictions.
    manual_items: list of dicts with keys image_path, label, rationale
    Returns list of corrected dicts with same order as input.
    """
    if not manual_items:
        return []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as in_f:
        json.dump(manual_items, in_f, indent=4)
        input_json = in_f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as out_f:
        output_json = out_f.name

    cmd = [
        sys.executable,
        str(correction_tool_path),
        "--input_json", input_json,
        "--output_json", output_json,
        "--label_map", *[str(c) for c in class_list],
    ]

    try:
        print(f"Launching manual correction tool: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        corrected = load_json(output_json)
        if not isinstance(corrected, list):
            raise ValueError("Manual correction output must be a list.")
        return corrected
    finally:
        try:
            os.remove(input_json)
        except Exception:
            pass
        try:
            os.remove(output_json)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Oracle script to correct prompts based on prompt bank.")
    parser.add_argument("--input_file", required=True, help="Path to the prompts.json file.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., 'Lurcher', 'FengRu').")
    parser.add_argument("--output_file", help="Path to save the corrected JSON. Defaults to overwriting input_file.")
    parser.add_argument(
        "--prompt_bank_path",
        default=None,
        help="Optional direct path to prompt bank JSON. If omitted, resolve from the local oracle directory.",
    )
    parser.add_argument(
        "--correction_tool_path",
        default=None,
        help="Optional path to apt_correction_tool_v2.py for manual batch correction.",
    )

    args = parser.parse_args()

    # Resolve repository paths relative to this script to avoid cwd-dependent bugs.
    apt_v3_dir = Path(__file__).resolve().parent
    repo_root = apt_v3_dir.parents[1]
    correction_tool_path = (
        Path(args.correction_tool_path)
        if args.correction_tool_path
        else repo_root / "apt_correction_tool_v2.py"
    )
    if args.prompt_bank_path:
        prompt_bank_path = Path(args.prompt_bank_path)
        print(f"Using prompt bank from --prompt_bank_path: {prompt_bank_path}")
    else:
        prompt_bank_path = resolve_prompt_bank_path(args.dataset, apt_v3_dir)

    class_list = None
    if "lurcher" in args.dataset.lower():
        class_list = ['wild', 'lurcher']

    # Load data
    prompts_data = load_json(args.input_file)
    prompt_bank_data = load_json(prompt_bank_path)
    
    # Create lookup for prompt bank (path -> explanation)
    # The prompt bank seems to be a dictionary with a "prompts" list or similar structure? 
    # Let's assume standard format based on inspection: {"prompts": [{"path":..., "explanation": ...}]}
    # Or it might be a direct list. Let's handle both or check inspection.
    # From inspection step 5: {"prompts": [{"path": ..., "class": ..., "explanation": [...]}, ...]}
    
    bank_lookup = {}
    if "prompts" in prompt_bank_data:
        bank_items = prompt_bank_data["prompts"]
    else:
        # Fallback if it's just a list
        bank_items = prompt_bank_data

    for item in bank_items:
        if "path" in item:
            bank_lookup[item["path"]] = item.get("explanation")

    if class_list is None:
        inferred_classes = sorted(
            {str(item.get("class")).lower() for item in bank_items if item.get("class") is not None}
        )
        class_list = inferred_classes
    if not class_list:
        print("Error: Could not infer class list from dataset/prompt bank.")
        sys.exit(1)

    
    if "prompts" not in prompts_data:
        print("Error: Input file must contain a 'prompts' key.")
        sys.exit(1)

    # Pre-process bank metadata for fallback search
    bank_metadata = []
    print("Pre-processing prompt bank for fallback searches...")
    for path, explanation in bank_lookup.items():
        meta = parse_image_name(path)
        if meta:
            meta["path"] = path # store path to retrieve data later
            bank_metadata.append(meta)

    updated_count = 0
    correct_count = 0
    wrong_count = 0
    total_images = 0
    
    # List to track images that could not be corrected
    uncorrected_images = []
    manual_correction_queue = []
    manual_queue_indices = []

    new_prompts_list = []

    # Process images
    for item in prompts_data["prompts"]:
        total_images += 1
        
        # Determine keys dynamically
        path_key = "image_path" if "image_path" in item else "path"
        class_key = "class" if "class" in item else "c"
        if "explanation" in item:
            exp_key = "explanation"
        elif "rationale" in item:
            exp_key = "rationale"
        else:
            exp_key = "e"

        # Get image path
        image_path = item.get(path_key)
        if not image_path:
            print(f"Warning: Item {item} has no image path. Skipping.")
            continue

        # Determine Ground Truth
        gt_class = get_ground_truth(image_path, class_list)
        if not gt_class:
            print(f"Error: Could not determine ground truth for {image_path}. Check class list.")
            sys.exit(1)

        current_class = item.get(class_key)
        
        # Check Prediction
        is_correct = (str(current_class).lower() == gt_class.lower()) if current_class is not None else False
        
        if is_correct:
            correct_count += 1
        else:
            wrong_count += 1

        # Create new item with normalized keys
        explanation_text = item.get(exp_key, "")
        new_item = {
            "explanation": explanation_text,
            "rationale": explanation_text,
            "class": current_class,
            "image_path": image_path,
            "manual_corrected": False,
        }
    
        # Correction Logic
        explanation_data = None
        fallback_source_path = None
        
        # Level 0: Try exact match first
        if image_path in bank_lookup:
            explanation_data = bank_lookup[image_path]
        else:
            # Try suffix match (still Level 0 effectively)
            found_key = None
            for bank_key in bank_lookup:
                if image_path.endswith(bank_key):
                    if len(image_path) == len(bank_key) or image_path[-(len(bank_key)+1)] in [os.sep, '/']:
                        found_key = bank_key
                        break
            
            if found_key:
                explanation_data = bank_lookup[found_key]
            else:
                # Fallback Logic
                # Parse current image name
                target_meta = parse_image_name(image_path)
                
                if target_meta:
                    # Level 1: Same Case, Slide, Section
                    for meta in bank_metadata:
                        if (meta["case_id"] == target_meta["case_id"] and 
                            meta["slide_num"] == target_meta["slide_num"] and 
                            meta["section_num"] == target_meta["section_num"]):
                            fallback_source_path = meta["path"]
                            explanation_data = bank_lookup[fallback_source_path]
                            print(f"Fallback L1 (Section match) for {os.path.basename(image_path)} using {os.path.basename(fallback_source_path)}")
                            break
                    
                    if not explanation_data:
                        # Level 2: Same Case, Slide
                        for meta in bank_metadata:
                            if (meta["case_id"] == target_meta["case_id"] and 
                                meta["slide_num"] == target_meta["slide_num"]):
                                fallback_source_path = meta["path"]
                                explanation_data = bank_lookup[fallback_source_path]
                                print(f"Fallback L2 (Slide match) for {os.path.basename(image_path)} using {os.path.basename(fallback_source_path)}")
                                break
                                
                    # Level 3 fallback intentionally removed.

        if explanation_data:
            # Helper to get first explanation string
            new_explanation = ""
            if isinstance(explanation_data, list) and len(explanation_data) > 0:
                new_explanation = explanation_data[0]
            elif isinstance(explanation_data, str):
                new_explanation = explanation_data
            
            if new_explanation:
                new_item["explanation"] = new_explanation
                new_item["rationale"] = new_explanation
                updated_count += 1
                
                # If wrong, update class as well
                if not is_correct:
                    new_item["class"] = gt_class.lower()
        else:
            print(f"Image not found in bank (even after fallback): {image_path}")
            uncorrected_item = {
                "image_path": image_path,
                "predicted_class": current_class,
                "predicted_explanation": explanation_text,
            }
            uncorrected_images.append(uncorrected_item)
            if not is_correct:
                # Queue only wrong predictions for one-shot manual correction.
                manual_correction_queue.append({
                    "image_path": image_path,
                    "label": str(current_class) if current_class is not None else "",
                    "rationale": explanation_text,
                })
                manual_queue_indices.append(len(new_prompts_list))
            
        new_prompts_list.append(new_item)

    # Batch manual correction for unresolved wrong predictions.
    if manual_correction_queue:
        if not correction_tool_path.exists():
            print(f"Warning: Manual correction tool not found: {correction_tool_path}")
        else:
            try:
                corrected_items = run_manual_corrections(
                    manual_correction_queue,
                    class_list,
                    correction_tool_path
                )
                if len(corrected_items) != len(manual_queue_indices):
                    print(
                        f"Warning: Manual correction output size mismatch "
                        f"(expected {len(manual_queue_indices)}, got {len(corrected_items)})."
                    )
                for idx, corrected in zip(manual_queue_indices, corrected_items):
                    corr_label = corrected.get("label", corrected.get("class"))
                    corr_rationale = corrected.get("rationale", corrected.get("explanation", ""))
                    if corr_label is not None:
                        new_prompts_list[idx]["class"] = str(corr_label)
                    if corr_rationale:
                        new_prompts_list[idx]["rationale"] = str(corr_rationale)
                        new_prompts_list[idx]["explanation"] = str(corr_rationale)
                    new_prompts_list[idx]["manual_corrected"] = True
                    updated_count += 1
            except Exception as e:
                print(f"Warning: Manual correction step failed: {e}")
            
    # Replace old list with new list
    prompts_data["prompts"] = new_prompts_list

    # Calculate statistics
    caption_correction_count = wrong_count
    try:
        # returns per-image time in minutes (float)
        per_image_time = estimate_annotation_time(correct_count, total_images)
        duration_minutes = per_image_time * total_images
    except Exception as e:
        print(f"Warning: Could not calculate duration: {e}")
        duration_minutes = 0.0

    # Calculate accuracy
    try:
        accuracy = round((correct_count / total_images) * 100, 1) if total_images > 0 else 0.0
    except Exception as e:
        print(f"Warning: Could not calculate accuracy: {e}")
        accuracy = 0.0

    # Update prompts_data with new stats
    prompts_data["caption correction count"] = caption_correction_count
    prompts_data["Duration in minutes"] = duration_minutes
    prompts_data["accuracy"] = accuracy
    
    if uncorrected_images:
        prompts_data["incorrect predictions"] = uncorrected_images
    else:
         pass

    # Summary
    print("-" * 30)
    print("Oracle Script Summary:")
    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Wrong Predictions: {wrong_count}")
    print(f"Updated Entries: {updated_count}")
    print(f"Caption Correction Count: {caption_correction_count}")
    print(f"Duration in Minutes: {duration_minutes}")
    print(f"Accuracy: {accuracy}%")
    print("-" * 30)

    # Save output
    output_path = args.output_file if args.output_file else args.input_file
    save_json(prompts_data, output_path)
    
    return accuracy

if __name__ == "__main__":
    main()
