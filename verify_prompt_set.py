import json
import os
import sys

# Mock code_mdl imports
# We just want to verify data loading and formatting
from utils import load_data

def verify():
    round2_prompt_set_path = "round-2-prompt_set.json"
    val_json_path = "datasets/microscopy_lurcher/fold-6/test.jsonl"
    
    print(f"Loading round 2 prompt set from {round2_prompt_set_path}...")
    with open(round2_prompt_set_path, 'r') as f:
        prompt_set_r2_dicts = json.load(f)
    
    print(f"Loaded {len(prompt_set_r2_dicts)} items.")
    
    # Simulate recover_round2.py logic (after user revert)
    prompt_set_r2 = []
    for item in prompt_set_r2_dicts:
        caption = f"{item['rationale']} C: {item['class']}"
        prompt_set_r2.append((item['image_path'], caption))
        
    print("\nFirst 3 items in prompt_set:")
    for i in range(3):
        print(f"Item {i}:")
        print(f"  Image: {prompt_set_r2[i][0]}")
        print(f"  Caption: {prompt_set_r2[i][1]}")
        
    print("\nLast 3 items in prompt_set:")
    for i in range(len(prompt_set_r2)-3, len(prompt_set_r2)):
        print(f"Item {i}:")
        print(f"  Image: {prompt_set_r2[i][0]}")
        print(f"  Caption: {prompt_set_r2[i][1]}")

    # Verify validation data loading
    print(f"\nLoading validation data from {val_json_path}...")
    known_labels = ['wild', 'lurcher']
    val_data = load_data(val_json_path, known_labels=known_labels)
    print(f"Loaded {len(val_data)} validation examples.")
    
    # Check class distribution
    labels = [y for _, y in val_data]
    from collections import Counter
    counts = Counter(labels)
    print(f"Class distribution: {counts}")
    print(f"0 (wild): {counts[0]}")
    print(f"1 (lurcher): {counts[1]}")
    
    # Verify map_label logic
    print("\nVerifying label mapping logic...")
    def map_label(label_str, known_labels):
        if label_str in [l.upper() for l in known_labels]:
            return known_labels.index(label_str.lower())
        return -1

    test_labels = ["WILD", "LURCHER", "wild", "lurcher"]
    for l in test_labels:
        mapped = map_label(l.upper(), known_labels)
        print(f"'{l}' -> {mapped}")

if __name__ == "__main__":
    verify()
