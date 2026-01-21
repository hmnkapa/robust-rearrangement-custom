from collections import deque
from typing import Dict, Tuple, Union, List
import torch
import torch.nn as nn
from omegaconf import DictConfig
import sys
import pathlib
import numpy as np

from src.behavior.base import Actor
from src.common.geometry import proprioceptive_quat_to_6d_rotation, quaternion_raw_multiply

# Add DP3 to path
DP3_PATH = pathlib.Path(__file__).parent.parent.parent / "third_party" / "3D-Diffusion-Policy" / "3D-Diffusion-Policy"
sys.path.append(str(DP3_PATH))

try:
    from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer as DP3LinearNormalizer
    from train import TrainDP3Workspace
except ImportError:
    print("Could not import DP3 modules. Make sure 3D-Diffusion-Policy submodule is initialized.")

from ipdb import set_trace as bp  # noqa

STATE_FIELDS: List[Tuple[str, ...]] = [
    ("robot_state", "joint_positions"),
    ("robot_state", "joint_velocities"),
    ("robot_state", "joint_torques"),
    ("robot_state", "ee_pos"),
    ("robot_state", "ee_quat"),
    ("robot_state", "ee_pos_vel"),
    ("robot_state", "ee_ori_vel"),
    ("robot_state", "gripper_width"),
]


class DP3Actor(Actor):
    """
    DP3 Actor that wraps the 3D-Diffusion-Policy model
    """
    
    def __init__(
        self,
        device: Union[str, torch.device], 
        cfg: DictConfig,
    ):
        # Initialize base actor - this sets up normalizer, observation queues, etc.
        super().__init__(device, cfg)
        
        # DP3-specific parameters from config
        actor_cfg = cfg.actor
        self.dp3_checkpoint_path = actor_cfg.checkpoint_path
        
        # Flag to tell rollout.py not to flatten robot_state
        self.expects_raw_robot_state = True
        
        # Load DP3 model from checkpoint
        self._load_dp3_model()
        
        # Override obs_dim calculation for point cloud input
        # DP3 uses agent_pos + point_cloud, so we need to recalculate
        self.agent_pos_dim = 37
        self.point_cloud_points = getattr(actor_cfg, 'point_cloud_points', 1024)
        self.point_cloud_dim = 3  # x, y, z coordinates
        
        # Update timestep obs dim for point cloud observation type
        if self.observation_type == "point_cloud":
            self.timestep_obs_dim = self.agent_pos_dim + self.point_cloud_points * self.point_cloud_dim
            self.obs_dim = (
                self.timestep_obs_dim * self.obs_horizon
                if self.flatten_obs
                else self.timestep_obs_dim
            )

    def _load_dp3_model(self):
        """Load DP3 model from checkpoint"""
        print(f"Loading DP3 model from: {self.dp3_checkpoint_path}")
        
        # Load the workspace from checkpoint
        # We need to map location to device to avoid CUDA errors if loading on different device
        payload = torch.load(open(self.dp3_checkpoint_path, 'rb'), pickle_module=sys.modules['dill'], map_location=self.device)
        cfg = payload['cfg']
        
        # Create workspace
        workspace = TrainDP3Workspace(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Extract the policy
        # In TrainDP3Workspace, the policy is stored in self.model
        # If EMA is used, we should use self.ema_model if available
        if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
            print("Using EMA model")
            ema_model = workspace.ema_model
            # Check if ema_model is the policy itself (has predict_action)
            if hasattr(ema_model, 'predict_action'):
                self.dp3_policy = ema_model
            elif hasattr(ema_model, 'model'):
                # EMA model might be wrapped in EMAModel class
                self.dp3_policy = ema_model.model
            else:
                self.dp3_policy = ema_model
        else:
            print("Using standard model (no EMA)")
            self.dp3_policy = workspace.model
            
        print(f"DP3 Policy type: {type(self.dp3_policy)}")
        
        self.dp3_policy.eval()
        
        # Load normalizer
        self.dp3_normalizer = None
        # Normalizer is usually part of the policy in DP3
        if hasattr(self.dp3_policy, 'normalizer'):
            self.dp3_normalizer = self.dp3_policy.normalizer
        elif hasattr(workspace, 'normalizer'):
            self.dp3_normalizer = workspace.normalizer
        else:
            print("Warning: Could not find normalizer in workspace or policy")
            # Try to find it in the payload directly if possible, or assume it's embedded in policy
            pass
        
        # Move to correct device
        self.dp3_policy = self.dp3_policy.to(self.device)
        # Normalizer usually doesn't have parameters but if it does:
        if self.dp3_normalizer is not None and hasattr(self.dp3_normalizer, 'to'):
            self.dp3_normalizer.to(self.device)
            
        # Alias for compatibility with rollout.py
        self.model = self.dp3_policy
        self.normalizer = self.dp3_normalizer
            
        print(f"DP3 model loaded successfully on device: {self.device}")

    def _normalized_obs(self, obs: deque, flatten: bool = True):
        """
        Construct and normalize observations for DP3
        """
        # 1. Construct agent_pos
        agent_pos_list = []
        point_cloud_list = []
        
        for o in obs:
            # Extract agent_pos fields
            step_agent_pos_parts = []
            for field_path in STATE_FIELDS:
                val = o
                for key in field_path:
                    if isinstance(val, dict):
                        val = val[key]
                    else:
                        # Fallback if val is not dict (should not happen if expects_raw_robot_state=True)
                        pass
                
                # Ensure val is tensor
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, device=self.device)
                
                # Handle dimensions
                if val.dim() == 1:
                    val = val.unsqueeze(0) # (1, dim)
                # Ensure batch dim matches
                if isinstance(o.get("robot_state"), dict):
                    # Assuming first field in robot_state has correct batch dim
                    pass
                     
                step_agent_pos_parts.append(val)
            
            step_agent_pos = torch.cat(step_agent_pos_parts, dim=-1)
            agent_pos_list.append(step_agent_pos)
            
            # Extract point cloud
            if "point_cloud" in o:
                pc = o["point_cloud"]
                if not isinstance(pc, torch.Tensor):
                    pc = torch.tensor(pc, device=self.device)
                point_cloud_list.append(pc)
            else:
                # Handle missing point cloud?
                # Should not happen if rollout is correct
                pass

        # Stack over time
        agent_pos = torch.stack(agent_pos_list, dim=1) # (B, T, D_pos)
        point_cloud = torch.stack(point_cloud_list, dim=1) # (B, T, N, 3)
        
        # Normalize
        # DP3 normalizer expects dict
        data = {
            "agent_pos": agent_pos,
            "point_cloud": point_cloud
        }
        
        # dp3 内部会 normalize
        # normalized_data = self.dp3_normalizer.normalize(data)
        
        return data

    def _normalized_action(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict actions using DP3
        """
        # DP3 policy expects dict with "agent_pos" and "point_cloud"
        # nobs is already the normalized dict
        
        # Run policy
        # DP3 policy forward expects batch
        # It returns dict with "action"
        
        result = self.dp3_policy.predict_action(nobs)
        action = result["action"]
        
        return action
        
    def _sample_action_pred(self, nobs):
        # Override to handle dict nobs
        # NOTE: _normalized_action returns UNNORMALIZED action from DP3 policy
        action = self._normalized_action(nobs)
        
        # Handle Action Alignment (Delta -> Absolute) if needed
        # Training data: 3D delta pos + 4D delta quat + 1D gripper
        # Env expects: Absolute Pos + Absolute Quat + Gripper (if action_type == "pos")
        
        if self.action_type == "pos":
            # Get current robot state from the last observation
            # self.observations is a deque of raw observations
            current_obs = self.observations[-1]
            rs = current_obs["robot_state"]
            
            # Extract current pos and quat
            # Shape: (B, 3) and (B, 4)
            curr_pos = rs["ee_pos"].to(self.device)
            curr_quat = rs["ee_quat"].to(self.device)
            
            # Action shape: (B, T, D)
            B, T, D = action.shape
            
            # Ensure current state has batch dim matching action
            if curr_pos.dim() == 1:
                curr_pos = curr_pos.unsqueeze(0) # (1, 3)
            if curr_quat.dim() == 1:
                curr_quat = curr_quat.unsqueeze(0) # (1, 4)
                
            # Expand to horizon
            # We assume the delta is relative to the CURRENT state for the whole horizon
            # (i.e. open-loop execution from current state)
            curr_pos_expanded = curr_pos.unsqueeze(1).expand(B, T, 3)
            curr_quat_expanded = curr_quat.unsqueeze(1).expand(B, T, 4)
            
            # Parse Action
            delta_pos = action[:, :, :3]
            delta_quat = action[:, :, 3:7]
            gripper = action[:, :, 7:]
            
            # 1. Apply Delta Pos
            # Target = Current + Delta
            target_pos = curr_pos_expanded + delta_pos
            
            # 2. Apply Delta Quat
            # Target = Current * Delta
            # Using src.common.geometry.quaternion_raw_multiply (x, y, z, w)
            target_quat = quaternion_raw_multiply(curr_quat_expanded, delta_quat)
            
            # Reassemble Action
            action = torch.cat([target_pos, target_quat, gripper], dim=-1)
        
        # Convert to deque of actions for the horizon
        actions = deque()
        # DP3 policy (SimpleDP3) already slices the output to start from the current step (t=0)
        # using internal logic `start = To - 1`. 
        # So we should always start from 0 here, regardless of predict_past_actions flag.
        start = 0 
        end = start + self.action_horizon
        
        # Ensure end does not exceed prediction length
        if end > action.shape[1]:
            end = action.shape[1]
            
        for i in range(start, end):
            actions.append(action[:, i, :])
            
        return actions