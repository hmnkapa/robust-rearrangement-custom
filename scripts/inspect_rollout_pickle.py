#!/usr/bin/env python3
import pickle
import numpy as np
import argparse
import os
import glob
from typing import Dict, Any

def inspect_pickle(pickle_path: str, max_steps: int):
    print(f"Inspecting pickle file: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # 1. METADATA
    print("\n" + "="*60)
    print("METADATA")
    print("="*60)
    
    # Explicitly requested fields
    print(f"success: {data.get('success', 'N/A')}")
    print(f"action_type: {data.get('action_type', 'N/A')}")
    
    sequence_keys = ['observations', 'actions', 'rewards', 'success', 'action_type']
    for key in sorted(data.keys()):
        if key not in sequence_keys:
             print(f"{key}: {data[key]}")
             
    # 2. SEQUENCE DATA
    observations = data.get('observations', [])
    actions = data.get('actions', [])
    rewards = data.get('rewards', [])
    
    n_obs = len(observations)
    n_act = len(actions)
    n_rew = len(rewards)
    
    print("\n" + "="*60)
    print(f"SEQUENCE DATA (obs={n_obs}, act={n_act}, rew={n_rew})")
    print(f"Displaying first {max_steps} steps...")
    print("="*60)
    
    limit = min(n_obs, max_steps)
    
    for i in range(limit):
        print(f"\n>>> Step {i} <<<")
        
        # --- Observation ---
        print("[Observation]:")
        if i < n_obs:
            def find_and_print_gripper(d, current_prefix=""):
                found = False
                if isinstance(d, dict):
                    for k, v in sorted(d.items()):
                        path = f"{current_prefix}/{k}" if current_prefix else k
                        if "gripper_width" in k:
                             print_val(v, prefix=f"  {path}")
                             found = True
                        elif isinstance(v, dict):
                             if find_and_print_gripper(v, path):
                                 found = True
                return found

            if not find_and_print_gripper(observations[i]):
                 print("  (gripper_width not found)")
        else:
            print("  (None)")
            
        # --- Action ---
        print("[Action]:")
        if i < n_act:
            # Actions might be list or array
            val = actions[i]
            print_val(val, prefix="  ")
        else:
            print("  (None)")

        # --- Reward ---
        if i < n_rew:
             val = rewards[i]
             print(f"[Reward]: {val}")

def recursive_inspect(data: Any, prefix: str = ""):
    if isinstance(data, dict):
        for key in sorted(data.keys()):
            value = data[key]
            new_prefix = f"{prefix}{key}"
            if isinstance(value, dict):
                print(f"{new_prefix}:")
                recursive_inspect(value, prefix=new_prefix + "/")
            else:
                print_val(value, new_prefix)
    else:
         print_val(data, prefix)

def print_val(value, prefix):
    if isinstance(value, np.ndarray) or (hasattr(value, 'shape') and hasattr(value, 'dtype')):
        # Handle numpy-like objects
        shape_str = f"shape={value.shape}"
        try:
            shape_str += f", dtype={value.dtype}"
        except:
            pass
            
        if value.size < 20: 
            # Small array: print simplified content
            try:
                content = str(value.flatten()) if hasattr(value, 'flatten') else str(value)
                # Remove newlines for cleaner output
                content = content.replace('\n', ' ')
                print(f"{prefix}: {content} ({shape_str})")
            except:
                 print(f"{prefix}: {shape_str} (errored printing content)")
        else:
            # Large array: print stats only (no raw data)
            if np.issubdtype(value.dtype, np.number):
                 try:
                     print(f"{prefix}: {shape_str} | min={np.min(value):.3f}, max={np.max(value):.3f}, mean={np.mean(value):.3f}")
                 except:
                     print(f"{prefix}: {shape_str}")
            else:
                 print(f"{prefix}: {shape_str}")
                 
    elif isinstance(value, (list, tuple)):
         # If list of numbers or small list, print content
         is_numeric = False
         if len(value) > 0 and isinstance(value[0], (int, float, np.number)):
             is_numeric = True
         
         if is_numeric or len(value) < 10:
             print(f"{prefix}: {value}")
         else:
             print(f"{prefix}: len={len(value)} (type={type(value)})")
         
    else:
         print(f"{prefix}: {value}")

def main():
    parser = argparse.ArgumentParser(description="Inspect a pickle file containing rollout data.")
    parser.add_argument("pickle_path", nargs='?', help="Path to the pickle file to inspect. If not provided, uses the latest file in the default directory.")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to display.")

    args = parser.parse_args()

    default_dir = "/data/hy/robust-rearrangement/raw/raw/diffik/sim/one_leg/rollout/low/pc/1024/fps/success/"
    
    pickle_path = args.pickle_path
    
    if pickle_path is None:
        # Find latest pickle in default directory
        if not os.path.exists(default_dir):
            print(f"Error: Default directory does not exist: {default_dir}")
            return
            
        list_of_files = glob.glob(os.path.join(default_dir, '*.pkl')) 
        if not list_of_files:
             print(f"Error: No pickle files found in {default_dir}")
             return
             
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"No path provided. Using latest file from default directory: {latest_file}")
        pickle_path = latest_file
    
    if not os.path.exists(pickle_path):
        print(f"Error: File not found: {pickle_path}")
        return

    inspect_pickle(pickle_path, args.steps)

if __name__ == "__main__":
    main()
