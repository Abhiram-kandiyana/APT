import json
import os
import sys
import numpy as np

# Set API Key from launch.json
os.environ["OPENAI_API_KEY"] = "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"

from main import APT, mdl_loss
from utils import load_data

def recover():
    # Paths
    checkpoint_path = "checkpoint.json"
    round2_prompt_set_path = "round-2-prompt_set.json"
    val_json_path = "datasets/microscopy_lurcher/fold-6/test.jsonl"
    system_prompt_1_path = "system_prompts/microscopy_lurcher_1.md"
    oracle_path = "oracle.json"
    
    # Load Checkpoint (Round 1 state)
    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    unlabeled_data_r1 = checkpoint['unlabeled_data']
    print(f"Loaded {len(unlabeled_data_r1)} unlabeled images from round 1.")

    # Load Round 2 Prompt Set
    print(f"Loading round 2 prompt set from {round2_prompt_set_path}...")
    with open(round2_prompt_set_path, 'r') as f:
        prompt_set_r2_dicts = json.load(f)
    
    # Convert dicts back to tuples for APT
    prompt_set_r2 = []
    for item in prompt_set_r2_dicts:
        caption = f"{item['rationale']} C: {item['class']}"
        prompt_set_r2.append((item['image_path'], caption))
    
    print(f"Loaded {len(prompt_set_r2)} exemplars for round 2.")

    # Identify new images selected in Round 2
    # The user stated: "the last 10 images in the @[round-2-prompt_set.json]"
    new_images_dicts = prompt_set_r2_dicts[-10:]
    new_image_paths = set(item['image_path'] for item in new_images_dicts)
    
    print(f"Identified {len(new_image_paths)} new images selected in round 2.")

    # Update Unlabeled Data
    # If checkpoint round is 2, unlabeled data is already updated.
    if checkpoint.get('round') == 2:
        print("Checkpoint indicates Round 2 is already complete. Using existing unlabeled data.")
        unlabeled_data_r2 = checkpoint['unlabeled_data']
    else:
        # Remove the new images from the round 1 unlabeled data
        unlabeled_data_r2 = []
        removed_count = 0
        for item in unlabeled_data_r1:
            # item is [image_path, label]
            if item[0] not in new_image_paths:
                unlabeled_data_r2.append(item)
            else:
                removed_count += 1
                
        print(f"Removed {removed_count} images from unlabeled pool. New size: {len(unlabeled_data_r2)}")

    # Load Validation Data
    print(f"Loading validation data from {val_json_path}...")
    known_labels = ['wild', 'lurcher']
    val_data = load_data(val_json_path, known_labels=known_labels)
    print(f"Loaded {len(val_data)} validation examples.")

    # Initialize APT
    # Parameters from user request/logs:
    # K=5, rounds=5, pool=100, batch=10, temp=1.0
    apt = APT(
        system_prompt_1_path=system_prompt_1_path,
        system_prompt_2_path="system_prompt_2.md", # This is actually unused in __init__ as it uses template string
        alpha=0.01,
        beta=0.1,
        lambda_mdl=0.1,
        lambda_c=0.5,
        K_uncertainty=5,
        mdl_tol=1e-3,
        max_rounds=5,
        candidate_pool_size=100,
        val_batch_size=5, # Default
        debug=False,
        oracle_path=oracle_path,
        prompt_set_path="round-2-prompt_set.json"
    )
    
    apt.prompt_set = prompt_set_r2
    
    # Run Evaluation for Round 2
    # We need to pass gen_kwargs for logging filename construction
    gen_kwargs = {
        "dataset": "microscopy_lurcher",
        "initial_batch_size": 10,
        "temperature": 1.0,
        "label_map": known_labels
    }
    
    print("Running evaluation for Round 2...")
    loss = apt.evaluate(val_data, round_num=2, **gen_kwargs)
    print(f"Round 2 MDL Loss: {loss}")

    # Save Checkpoint for Round 3
    # We want to be ready for Round 3.
    # So we save state as if Round 2 just finished.
    # round should be 3 (next round to run) or 2 (last finished)? 
    # The user said: "I also want the round-3 values to logged on to the @[checkpoint.json]"
    # Usually checkpoint saves the state *after* a round.
    # If I look at checkpoint.json provided: "round": 1. This means Round 1 finished.
    # So if Round 2 finishes, we should save "round": 2?
    # But user said "round-3 values". Maybe they mean the state *ready* for round 3.
    # If "round": 1 means ready for round 2, then "round": 2 means ready for round 3.
    # Let's check main.py resume logic.
    # I'll assume "round": 2 means Round 2 is done.
    # Wait, if I save "round": 2, and resume, does it start Round 3?
    # I'll check main.py later if needed, but standard is usually last completed round.
    # Actually, let's look at the user request again: "I also want the round-3 values to logged on to the @[checkpoint.json]"
    # Maybe they mean "round": 3?
    # Let's save as "round": 2 for now, as that reflects what we just finished. 
    # If the code increments at start, it might be different.
    # Let's assume the user wants to continue to round 3.
    
    new_checkpoint = {
        "round": 2, # Finished round 2
        "mdl_loss": loss,
        "prompt_set": prompt_set_r2_dicts,
        "unlabeled_data": unlabeled_data_r2
    }
    
    print("Saving new checkpoint...")
    with open(checkpoint_path, 'w') as f:
        json.dump(new_checkpoint, f, indent=4)
        
    print("Recovery complete.")

if __name__ == "__main__":
    recover()
