import torch
from omegaconf import OmegaConf
import argparse
import os
import sys

def create_dp3_wrapper(dp3_ckpt_path, output_path):
    # Construct Config
    cfg = OmegaConf.create({
        "actor_name": "dp3",
        "observation_type": "point_cloud",
        "robot_state_dim": 14, # Kept as default, DP3Actor will ignore this and use agent_pos_dim
        "action_dim": 8, # 7 (pose) + 1 (gripper)
        "actor": {
            "name": "dp3",
            "checkpoint_path": os.path.abspath(dp3_ckpt_path),
            "point_cloud_points": 1024,
            "obs_horizon": 2,
            "pred_horizon": 16,
            "action_horizon": 8,
            "predict_past_actions": False,
            "flatten_obs": False
        },
        "control": {
            "control_mode": "delta", # Ensure consistency with training
            "act_rot_repr": "quat", # or rot_6d, depending on training
            "rot_repr": "quat" # Added as requested
        },
        "data": {
            "augment_image": False
        },
        "regularization": {},
        "discount": 0.99
    })

    # Save Checkpoint
    # We save the config in the checkpoint so evaluate_model.py can load it
    # The state_dict is empty because DP3Actor loads its own weights from checkpoint_path
    torch.save({"config": cfg, "state_dict": {}}, output_path)
    print(f"Saved wrapper checkpoint to {output_path}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a wrapper checkpoint for DP3 evaluation")
    parser.add_argument("--dp3-ckpt", type=str, required=True, help="Path to the trained DP3 checkpoint (.ckpt)")
    parser.add_argument("--output", type=str, default="dp3_wrapper.pt", help="Output path for the wrapper checkpoint")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dp3_ckpt):
        print(f"Error: DP3 checkpoint not found at {args.dp3_ckpt}")
        sys.exit(1)
        
    create_dp3_wrapper(args.dp3_ckpt, args.output)
