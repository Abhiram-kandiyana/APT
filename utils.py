import re
import json
import os
from pathlib import Path

def _extract_label_after_c(answer_text, known_labels=None):
    """
    Extract the label token that follows `C:` in microscopy prompts/answers.

    Args:
        answer_text: The raw answer or rationale string.
        known_labels: Optional iterable of allowed labels to validate against.
    """
    if answer_text is None:
        return ""

    text = str(answer_text).lower()
    # Allow variations like "c:", "- c:", "'c':", "**c**:", "- **c**:", etc.
    match = re.search(r'[c][\s\*\'\"_-]*:\s*([^\n\r]+)', text, flags=re.IGNORECASE)
    if match:
        raw_after = match.group(1).strip()
        # Strip leading markdown/punctuation before tokenizing
        raw_after = raw_after.lstrip("*'\"`_- ")
        # Tokenize and take the first non-empty token
        candidate_token = ""
        for tok in raw_after.split():
            cleaned = tok.strip(".,;:*\"'`_")
            if cleaned != "":
                candidate_token = cleaned
                break
        candidate = candidate_token
    else:
        tokens = text.strip().split()
        candidate = tokens[-1] if tokens else ""

    # strip leading/trailing punctuation/markdown such as '.', ',', '*', quotes
    candidate = candidate.strip(".,;:*\"'`_")

    if known_labels:
        for lbl in known_labels:
            if candidate == lbl.lower():
                return candidate
        return ""
    return candidate

def load_data(json_path, known_labels=None):
    data = []
    decoder = json.JSONDecoder()

    with open(json_path, encoding="utf-8") as f:
        for line in f:
            json_res = decoder.raw_decode(line)[0]
            image_path = str(json_res.get("image_path", "")).strip()
            caption = json_res.get("class", "")
            label_str = _extract_label_after_c(caption, known_labels=known_labels)
            # Convert label string to index in known_labels
            if known_labels and label_str:
                try:
                    label = known_labels.index(label_str.lower())
                except ValueError:
                    # Label not found in known_labels
                    label = -1
            else:
                label = -1
            data.append((image_path, label))
    return data
def load_cached_preds(gen_kwargs):
    cached_preds = {}

    vlm_log_path = gen_kwargs.get("vlm_log_path")
    if vlm_log_path and os.path.exists(vlm_log_path):
        try:
            with open(vlm_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    log_data = json.loads(content)
                    # Create a map from image_path to dict
                    for entry in log_data:
                        img_path = entry.get("image_path")
                        lbl = entry.get("label")
                        rationale = entry.get("rationale", "")
                        if img_path is not None:
                            cached_preds[str(img_path)] = {
                                "label": lbl,
                                "rationale": rationale
                            }
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read VLM log from {vlm_log_path}: {e}")
    return cached_preds


def create_split_jsonl_files(splits_json_path: str, train_filename: str = "train.jsonl",
                             test_filename: str = "test.jsonl"):
    """
    Convert a microscopy splits.json file into newline-delimited JSON for train and test splits.

    Args:
        splits_json_path: Path to the splits.json file containing "train" and "test" keys.
        train_filename: Name of the train jsonl to create alongside the splits file.
        test_filename: Name of the test jsonl to create alongside the splits file.
    """
    splits_path = Path(splits_json_path)
    output_dir = splits_path.parent

    with splits_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)

    missing_keys = [key for key in ("train", "test") if key not in splits]
    if missing_keys:
        raise ValueError(f"Missing split(s) in {splits_json_path}: {', '.join(missing_keys)}")

    def parse_entry(entry):
        if isinstance(entry, (list, tuple)):
            if len(entry) < 3:
                raise ValueError(f"Split entry has insufficient fields: {entry}")
            return str(entry[0]), str(entry[2])
        if isinstance(entry, dict):
            path_val = entry.get("image_path")
            label_val = entry.get("class")
            if path_val is None or label_val is None:
                raise ValueError(f"Split entry is missing path/label information: {entry}")
            return str(path_val), str(label_val)
        raise TypeError(f"Unsupported split entry type: {type(entry)}")

    def write_split(split_name, filename):
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as out_file:
            for entry in splits[split_name]:
                image_path, label = parse_entry(entry)
                record = {"image_path": image_path, "class": label}
                out_file.write(json.dumps(record) + "\n")
        return output_path

    train_path = write_split("train", train_filename)
    test_path = write_split("test", test_filename)
    return str(train_path), str(test_path)

if __name__ == "__main__":
    create_split_jsonl_files("datasets/microscopy_lurcher/fold-6/splits.json")
