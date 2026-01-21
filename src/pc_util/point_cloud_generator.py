import math
from typing import Iterable, List, Optional

import torch
from isaacgym import gymapi, gymtorch

# Optional visualization via Open3D
try:
    import open3d as o3d
except Exception:
    o3d = None

# Optional PyTorch3D FPS
try:
    from pytorch3d.ops import sample_farthest_points as _torch3d_sample_fps
except Exception:
    _torch3d_sample_fps = None

BBOX_Z_OFFSET = 0.1
FIXED_SCENE_X_THRESHOLD = 0.15  # For "fixed-scene" mode: keep points with x > base_pos[0] + threshold
FIXED_SCENE_Z_THRESHOLD = 0.25   # For "fixed-scene" mode: keep points with z < base_pos[2] + threshold
DEBUG_VISUALIZE = False

class PointCloudGenerator:
    """Generate point clouds from an Isaac Gym env camera and map to robot base frame."""

    def __init__(
        self,
        env,
        camera_name: str = "front",
        target_frame: str = "robot",
        max_points: int = 4096,
        bbox_half_extent = 0.2,
        bbox_crop_mode: str = "eepose-centered", # "eepose-centered" or "fixed-scene"
    ):
        self.env = env
        self.camera_name = camera_name
        self.target_frame = target_frame
        self.max_points = max_points
        self.device = getattr(env, "device", torch.device("cpu"))
        # Optional axis-aligned 3D bbox cropping centered at EE pose
        # If set to a scalar, uses the same half-extent for x/y/z; if an iterable of len 3, uses per-axis.
        self.bbox_half_extent = bbox_half_extent
        self.bbox_crop_mode = bbox_crop_mode
        # Tracks which EE pose source was used last (for debug)
        self._last_eepose_source: Optional[str] = None

        # Open3D visualizer state
        self._viz = None
        self._pcd = None
        self._warned_no_o3d = False

        # Ensure camera tensors are available for depth/segmentation
        if not hasattr(self.env, "camera_handles"):
            raise RuntimeError("Env must expose camera_handles; call set_camera first.")

    def _intrinsics(self):
        width, height = self.env.img_size
        fov = getattr(self.env, "camera_cfg", None)
        fov = fov.horizontal_fov if fov is not None else 69.4
        fov_rad = math.radians(float(fov))
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        )
        return K

    def _extrinsics_cam_to_sim(self):
        """Build camera-to-sim-local transform using a stable right-handed look-at basis.

        Returns transform from camera frame to Sim Local (environment-relative) frame.
        - forward (+Z camera) = cam_target - cam_pos
        - right = cross(forward, up_ref)
        - up    = cross(right, forward)
        Uses env.front_cam_up if provided; falls back to +Y, then +Z if colinear.
        """
        if not hasattr(self.env, "front_cam_pos") or not hasattr(
            self.env, "front_cam_target"
        ):
            raise RuntimeError("Front camera pose not cached on env.")

        cam_pos = torch.tensor(self.env.front_cam_pos, device=self.device, dtype=torch.float32)
        cam_target = torch.tensor(
            self.env.front_cam_target, device=self.device, dtype=torch.float32
        )
        forward = cam_target - cam_pos
        forward = forward / (forward.norm() + 1e-8)

        up_ref = getattr(self.env, "front_cam_up", None)
        if up_ref is None:
            # Default to Z-up for this Z-up sim environment
            up_ref = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        else:
            up_ref = torch.tensor(up_ref, device=self.device, dtype=torch.float32)

        right = torch.cross(forward, up_ref)
        if right.norm() < 1e-6:
            # If forward is colinear with Z (looking straight up/down), try Y
            up_ref = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            right = torch.cross(forward, up_ref)
        right = right / (right.norm() + 1e-8)
        up = torch.cross(right, forward)

        R = torch.stack([right, up, forward], dim=1)
        T = torch.eye(4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = cam_pos
        return T

    def _fetch_depth_and_seg(self, env_idx: int):
        handles = self.env.camera_handles.get(self.camera_name, None)
        if handles is None or len(handles) <= env_idx:
            raise RuntimeError(f"Camera handles missing for {self.camera_name}")
        handle = handles[env_idx]
        depth_tensor = gymtorch.wrap_tensor(
            self.env.isaac_gym.get_camera_image_gpu_tensor(
                self.env.sim, self.env.envs[env_idx], handle, gymapi.IMAGE_DEPTH
            )
        )
        # Segmentation may be unavailable; try and fall back gracefully
        seg_tensor = None
        try:
            seg_tensor = gymtorch.wrap_tensor(
                self.env.isaac_gym.get_camera_image_gpu_tensor(
                    self.env.sim,
                    self.env.envs[env_idx],
                    handle,
                    gymapi.IMAGE_SEGMENTATION,
                )
            )
        except Exception:
            seg_tensor = None
        return depth_tensor, seg_tensor

    def _unproject(self, depth: torch.Tensor, K: torch.Tensor, mask: torch.Tensor, env_idx: int = 0):
        """Unproject depth pixels back to Sim Local coordinates.
        
        Steps:
        1. Camera pixels → Camera 3D coordinates (using intrinsics K)
        2. Camera → Sim Local coordinates (using extrinsics)
        
        Returns points in Sim Local (environment-relative) coordinate frame.
        """
        # Intrinsics
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Grid
        H, W = depth.shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        
        # Mask selection and unprojection
        # depth is positive planar distance
        z_cam = depth[mask].float() 
        u = xs[mask].float()
        v = ys[mask].float()
        
        # Back-project to CV frame (X right, Y down, Z forward)
        x_cv = (u - cx) * z_cam / fx
        y_cv = (v - cy) * z_cam / fy
        
        # Convert to our Internal Cam Frame (X right, Y up, Z forward)
        # Based on _project_world_to_image: y_cv = -y_cam
        x_cam = x_cv
        y_cam = -y_cv
        
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1) # (N, 3)
        
        # Step 2: Camera → Sim Local (environment-relative coordinates)
        T_c2s = self._extrinsics_cam_to_sim()
        ones = torch.ones((pts_cam.shape[0], 1), device=self.device)
        pts_cam_h = torch.cat([pts_cam, ones], dim=-1)
        pts_sim_h = (T_c2s @ pts_cam_h.t()).t()
        pts_sim_local = pts_sim_h[:, :3]
        
        # Return Sim Local coordinates directly (no conversion to World Global)
        return pts_sim_local
    
    def _transform(self, pts_sim_local: torch.Tensor, target_frame: Optional[str] = None, env_idx: int = 0):
        """Transform Sim Local points to the requested target frame.

        Args:
            pts_sim_local: Points in Sim Local (environment-relative) coordinate frame
            target_frame: "world" or "robot". Defaults to `self.target_frame`.
            env_idx: Environment index
            
        Assumes robot base has no rotation, only translation (consistent with Furniture Bench).
        """
        if target_frame is None:
            target_frame = self.target_frame
        if target_frame == "robot":
            # Get robot base position in Sim Local frame
            if not hasattr(self.env, "rb_states") or not hasattr(self.env, "base_idxs"):
                raise RuntimeError("Env must have rb_states and base_idxs for robot frame transform.")
            # rb_states is already in Sim Local coordinates
            base_pos_sim_local = self.env.rb_states[self.env.base_idxs[env_idx], :3].detach().to(self.device).clone()
            # Transform to robot frame: subtract base position (assume no rotation)
            pts_robot = pts_sim_local - base_pos_sim_local
            return pts_robot
        return pts_sim_local

    def _downsample(self, pts: torch.Tensor, max_points: Optional[int], mode: str = "random"):
        """Downsample points to at most max_points.

        mode:
        - random: random permutation sampling
        - uniform: evenly spaced indices across the array order
        - fps: farthest point sampling based on Euclidean distance
        """
        if max_points is None:
            max_points = self.max_points
        n = pts.shape[0]
        
        # Pad with zeros if fewer points than max_points
        if n < max_points:
            padding = torch.zeros((max_points - n, 3), device=pts.device, dtype=pts.dtype)
            return torch.cat([pts, padding], dim=0)

        if n == max_points:
            return pts

        if mode == "uniform":
            # Evenly spaced indices from [0, n-1]
            idx = torch.linspace(0, n - 1, steps=max_points, device=pts.device)
            idx = idx.round().to(torch.long)
        elif mode == "fps":
            # Use PyTorch3D farthest point sampling exclusively
            if _torch3d_sample_fps is None:
                raise RuntimeError("PyTorch3D not available: install pytorch3d to use downsample_mode='fps'.")
            pb = pts[None, :, :].to(torch.float32)
            K_t = torch.as_tensor([max_points], device=pb.device)
            _, sampled_indices = _torch3d_sample_fps(points=pb[..., :3], K=K_t)
            idx = sampled_indices.squeeze(0).to(torch.long)
        else:
            idx = torch.randperm(n, device=pts.device)[:max_points]
        return pts[idx]

    def _visualize_points(self, pts: torch.Tensor):
        """Show/update interactive point cloud window with draggable view."""
        if o3d is None:
            if not self._warned_no_o3d:
                print("[PointCloudGenerator] Open3D not available; visualization disabled.")
                self._warned_no_o3d = True
            return

        pts_np = pts.detach().cpu().numpy()

        if self._viz is None:
            self._viz = o3d.visualization.Visualizer()
            self._viz.create_window(window_name="PointCloud Viewer", width=960, height=720)
            self._pcd = o3d.geometry.PointCloud()
            self._pcd.points = o3d.utility.Vector3dVector(pts_np)
            self._viz.add_geometry(self._pcd)
            render_opt = self._viz.get_render_option()
            render_opt.point_size = 2.0
        else:
            self._pcd.points = o3d.utility.Vector3dVector(pts_np)
            self._viz.update_geometry(self._pcd)

        self._viz.poll_events()
        self._viz.update_renderer()

    def _visualize_depth(self, depth: torch.Tensor, env_idx: int):
        """Visualize depth image alongside RGB image for debugging."""
        try:
            import cv2
            import numpy as np
            
            # Convert depth to numpy and normalize
            depth_np = depth.detach().cpu().numpy()
            depth_vis = depth_np.copy()
            
            # Normalize to 0-255 range
            depth_min = depth_vis.min()
            depth_max = depth_vis.max()
            if depth_max > depth_min:
                depth_vis = ((depth_vis - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)
            
            # Apply colormap
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Get RGB image from camera_obs if available
            color_key = "color_image2"  # front camera
            if color_key in self.env.camera_obs and len(self.env.camera_obs[color_key]) > env_idx:
                # Render first to get latest RGB
                self.env.isaac_gym.render_all_camera_sensors(self.env.sim)
                self.env.isaac_gym.start_access_image_tensors(self.env.sim)
                
                rgb_tensor = self.env.camera_obs[color_key][env_idx]
                rgb_np = rgb_tensor.detach().cpu().numpy()
                
                self.env.isaac_gym.end_access_image_tensors(self.env.sim)
                
                # Convert RGB to BGR for OpenCV (assume RGB format)
                if rgb_np.shape[-1] == 4:  # RGBA
                    rgb_np = rgb_np[..., :3]
                rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                
                # Resize to match depth if needed
                if rgb_bgr.shape[:2] != depth_color.shape[:2]:
                    rgb_bgr = cv2.resize(rgb_bgr, (depth_color.shape[1], depth_color.shape[0]))
                
                # Concatenate horizontally
                combined = np.hstack([rgb_bgr, depth_color])
                
                # Add text overlay
                text1 = f"Env {env_idx} - RGB"
                text2 = f"Depth: [{depth_min:.3f}, {depth_max:.3f}]"
                cv2.putText(combined, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                cv2.putText(combined, text2, (rgb_bgr.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                # Only depth available
                combined = depth_color
                text = f"Env {env_idx} Depth: min={depth_min:.3f}, max={depth_max:.3f}"
                cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
            
            # Show in separate window per env
            window_name = f"Env {env_idx}: RGB + Depth"
            cv2.imshow(window_name, combined)
            cv2.waitKey(1)  # Non-blocking
        except ImportError:
            print("[PointCloudGenerator] cv2 not available for depth visualization")
        except Exception as e:
            print(f"[PointCloudGenerator] Visualization error: {e}")

    def _get_ee_pose_sim_local(self, env_idx: int = 0) -> torch.Tensor:
        """Get EE pose center (position) in Sim Local frame.

        Reads from rb_states (World Global) and converts to Sim Local (environment-relative).
        Returns the EE position as a 3D tensor in Sim Local frame.
        """
        if not hasattr(self.env, "rb_states") or not hasattr(self.env, "ee_idxs"):
            raise RuntimeError("Env must have rb_states and ee_idxs for EE pose.")
        # rb_states: (num_rigid_bodies, 13) in World Global space
        # [:3] is position, [3:7] is quaternion
        
        # self.env.ee_idxs is a list of global indices, one per env.
        idx = self.env.ee_idxs[env_idx]
        # rb_states contains positions in SIM LOCAL coordinates (already relative to env_origin)
        pos_sim_local = self.env.rb_states[idx, :3].detach().to(self.device).float().clone()
        
        self._last_eepose_source = "rb_states:already_sim_local"
        return pos_sim_local

    def _apply_3d_bbox_crop(self, depth: torch.Tensor, half_extent, env_idx: int = 0) -> torch.Tensor:
        """Compute a boolean mask of pixels whose 3D points lie inside an axis-aligned bbox.

        The bbox is centered at the EE pose. This function computes its own validity mask
        based on positive, finite depth and does NOT require the caller's mask.
        """
        if half_extent is None:
            # If no bbox, keep everything (let caller intersect as needed)
            return torch.ones_like(depth, dtype=torch.bool)

        base_mask = torch.isfinite(depth) & (depth > 0)
        K = self._intrinsics()
        # _unproject returns points in Sim Local frame
        pts_sim_local = self._unproject(depth, K, base_mask, env_idx=env_idx)

        # Half extents
        if isinstance(half_extent, (int, float)):
            hx, hy, hz = float(half_extent), float(half_extent), float(half_extent)
        else:
            he = list(half_extent)
            if len(he) != 3:
                raise ValueError("bbox_half_extent must be a scalar or a length-3 iterable")
            hx, hy, hz = float(he[0]), float(he[1]), float(he[2])

        # Get EE position in Sim Local frame (same frame as pts_sim_local)
        ee_pos_sim_local = self._get_ee_pose_sim_local(env_idx=env_idx)
        # Shift bbox down by BBOX_Z_OFFSET
        center_sim_local = ee_pos_sim_local.clone()
        center_sim_local[..., 2] -= BBOX_Z_OFFSET

        lower = center_sim_local - torch.tensor([hx, hy, hz], device=self.device, dtype=torch.float32)
        upper = center_sim_local + torch.tensor([hx, hy, hz], device=self.device, dtype=torch.float32)

        keep = (pts_sim_local >= lower).all(dim=-1) & (pts_sim_local <= upper).all(dim=-1)

        # Map kept points back to full-resolution mask
        ys, xs = torch.where(base_mask)
        bbox_mask = torch.zeros_like(base_mask, dtype=torch.bool)
        if keep.numel() > 0:
            bbox_mask[ys[keep], xs[keep]] = True
        # Save for visualization
        self._last_bbox_mask = bbox_mask
        return bbox_mask

    def _apply_fixed_scene_crop(self, depth: torch.Tensor, env_idx: int = 0) -> torch.Tensor:
        """Keep points with x > base_pos[0] + FIXED_SCENE_X_THRESHOLD and z < base_pos[2] + FIXED_SCENE_Z_THRESHOLD."""
        base_mask = torch.isfinite(depth) & (depth > 0)
        K = self._intrinsics()
        # _unproject returns points in Sim Local frame
        pts_sim_local = self._unproject(depth, K, base_mask, env_idx=env_idx)
        
        if not hasattr(self.env, "rb_states") or not hasattr(self.env, "base_idxs"):
            base_pos_x = 0.0
            base_pos_z = 0.0
        else:
            idx = self.env.base_idxs[env_idx]
            base_pos = self.env.rb_states[idx, :3]
            base_pos_x = base_pos[0].item()
            base_pos_z = base_pos[2].item()
            
        # x direction crop
        keep_x = pts_sim_local[:, 0] > (base_pos_x + FIXED_SCENE_X_THRESHOLD)
        # z direction crop (filter out points too high, e.g. gripper or top camera artifacts)
        keep_z = pts_sim_local[:, 2] < (base_pos_z + FIXED_SCENE_Z_THRESHOLD)
        
        keep = keep_x & keep_z

        # Map kept points back to full-resolution mask
        ys, xs = torch.where(base_mask)
        bbox_mask = torch.zeros_like(base_mask, dtype=torch.bool)
        if keep.numel() > 0:
            bbox_mask[ys[keep], xs[keep]] = True
        self._last_bbox_mask = bbox_mask
        return bbox_mask

    def _project_sim_local_to_image(self, pts_sim_local):
        """Project Sim Local points to image pixels using internal K and extrinsics.
        
        Args:
            pts_sim_local: Points in Sim Local (environment-relative) coordinate frame
        """
        K = self._intrinsics()
        T_c2s = self._extrinsics_cam_to_sim()
        T_s2c = torch.inverse(T_c2s)

        # Transform from Sim Local to camera frame
        ones = torch.ones((pts_sim_local.shape[0], 1), device=pts_sim_local.device)
        pts_h = torch.cat([pts_sim_local, ones], dim=-1)
        pts_cam = (T_s2c @ pts_h.t()).t()[:, :3]

        # Convert to standard CV frame: (Right, Down, Forward)
        # Our extrinsics should build (Right=+X_cam, Up=+Y_cam, Forward=+Z_cam) basis.
        # CV camera frame expects +X_cv=Right, +Y_cv=Down, +Z_cv=Forward.
        # Since our "Up" (+Y_cam) points physically Up, and CV "Down" (+Y_cv) points physically Down,
        # we MUST negate Y.
        # Our "Right" (+X_cam) points physically Right (if constructed correctly with Z-up world).
        # CV "Right" (+X_cv) points physically Right.
        # So X should NOT be negated.
        
        pts_cv = pts_cam.clone()
        # pts_cv[:, 0] = pts_cv[:, 0] # Keep X (Right)
        pts_cv[:, 1] = -pts_cv[:, 1] # Flip Y (Up -> Down)

        # Project: p_img = K * p_cv
        pts_img = (K @ pts_cv.t()).t()
        
        u = pts_img[:, 0] / (pts_img[:, 2] + 1e-8)
        v = pts_img[:, 1] / (pts_img[:, 2] + 1e-8)
        return torch.stack([u, v], dim=-1)

    def _visualize_bbox_crop_on_image(self, env_idx: int, bbox_mask: torch.Tensor):
        """Fetch color image and show it with cropped-out pixels colored black.

        Minimal logic: get `IMAGE_COLOR`, apply mask, display via OpenCV.
        """
        import numpy as np
        handles = self.env.camera_handles.get(self.camera_name, None)
        if handles is None or len(handles) <= env_idx:
            print("[PointCloudGenerator] No camera handle for color visualization.")
            return
        handle = handles[env_idx]
        # Access color image
        self.env.isaac_gym.start_access_image_tensors(self.env.sim)
        color = gymtorch.wrap_tensor(
            self.env.isaac_gym.get_camera_image_gpu_tensor(
                self.env.sim, self.env.envs[env_idx], handle, gymapi.IMAGE_COLOR
            )
        )
        self.env.isaac_gym.end_access_image_tensors(self.env.sim)

        img = color.detach().cpu()
        # Expect HxWxC; reduce to RGB if RGBA
        if img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3]
        elif img.ndim == 3 and img.shape[0] >= 3:
            img = img[:3, ...].permute(1, 2, 0)
        # Convert to uint8 if needed (assume [0,1] or [0,255])
        if img.dtype != torch.uint8:
            maxv = float(img.max().item())
            if maxv <= 1.0 + 1e-6:
                img = (img * 255.0).clamp(0, 255)
            img = img.to(torch.uint8)
        img_np = img.numpy().copy()

        bm = bbox_mask.detach().cpu().numpy()
        img_np[~bm] = 0

        # Draw EE Pose and BBox
        if self.bbox_half_extent is not None:
            # handle list vs scalar
            if isinstance(self.bbox_half_extent, (int, float)):
                hx = hy = hz = float(self.bbox_half_extent)
            else:
                he = list(self.bbox_half_extent)
                hx, hy, hz = float(he[0]), float(he[1]), float(he[2])
            
            # Get EE position directly in Sim Local frame
            center_sim_local = self._get_ee_pose_sim_local(env_idx=env_idx) # (3,) in Sim Local
             
            # Shift bbox center down by BBOX_Z_OFFSET for box corners, but keep EE center for dot
            bbox_center_sim_local = center_sim_local.clone()
            bbox_center_sim_local[..., 2] -= BBOX_Z_OFFSET
             
            # 8 corners
            offsets = torch.tensor([
                [-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, -hz], [-hx, hy, hz],
                [hx, -hy, -hz],  [hx, -hy, hz],  [hx, hy, -hz],  [hx, hy, hz]
            ], device=self.device)
             
            # Generate corners based on shifted BBox center (in Sim Local frame)
            corners_sim_local = bbox_center_sim_local + offsets
             
            # Project (pts should be in Sim Local Frame to match Camera Extrinsics)
            # We project the original EE center (for the red dot) and the shifted BBox corners (for the green box)
            pts_to_proj = torch.cat([center_sim_local.unsqueeze(0), corners_sim_local], dim=0)
            uvs = self._project_sim_local_to_image(pts_to_proj)
            uvs_np = uvs.detach().cpu().numpy().astype(int)
            
            center_uv = uvs_np[0]
            corners_uv = uvs_np[1:]

            try:
                import cv2
                # Draw Center
                cv2.circle(img_np, tuple(center_uv), 5, (0, 0, 255), -1)
                
                # Draw Box Lines
                pairs = [
                    (0,1), (2,3), (4,5), (6,7),
                    (0,2), (1,3), (4,6), (5,7),
                    (0,4), (1,5), (2,6), (3,7)
                ]
                # Green box
                color_box = (0, 255, 0)
                for i, j in pairs:
                    pt1 = tuple(corners_uv[i])
                    pt2 = tuple(corners_uv[j])
                    cv2.line(img_np, pt1, pt2, color_box, 1)

                win = f"BBox Crop - {self.camera_name}"
                cv2.imshow(win, img_np[..., ::-1])
                cv2.waitKey(1)
            except Exception:
                print("[PointCloudGenerator] OpenCV not available for display.")
    
    def _crop(self, depth, seg, env_idx: int = 0):
        # Depth mask: valid >0 and finite
        valid = torch.isfinite(depth) & (depth > 0)
        valid_depth_count = int(valid.sum().item())

        # 主逻辑
        if seg is not None:
            allowed = (seg >= 5) | (seg == 4)
            valid = valid & allowed
        # Optional 3D bbox crop 
        if self.bbox_crop_mode == "fixed-scene":
             bbox_mask = self._apply_fixed_scene_crop(depth, env_idx=env_idx)
             valid = valid & bbox_mask
        elif self.bbox_half_extent is not None:
            bbox_mask = self._apply_3d_bbox_crop(depth, self.bbox_half_extent, env_idx=env_idx)
            valid = valid & bbox_mask
        valid_after_seg = int(valid.sum().item())

        # debug output
        if valid_depth_count == 0 or valid_after_seg == 0:
            depth_stats = (
                float(depth.min().item()),
                float(depth.max().item()),
                float(torch.isfinite(depth).float().mean().item()),
            )
            seg_stats = None
            if seg is not None:
                unique_seg = torch.unique(seg).cpu().tolist()
                seg_stats = {
                    "unique_ids": unique_seg[:10],
                    "num_unique": len(unique_seg),
                }
            print(
                f"[PointCloudGenerator] camera={self.camera_name}"
                f" depth_valid={valid_depth_count} after_seg={valid_after_seg}"
                f" depth_shape={tuple(depth.shape)} depth_minmax={depth_stats[:2]}"
                f" depth_finite_frac={depth_stats[2]:.3f} seg_stats={seg_stats}"
            )
            raise ValueError("No Valid Point")
        return valid

    def generate_transformed_cropped_point_cloud(
        self,
        env_idx: int = 0,
        max_points: Optional[int] = None,
        downsample_mode: str = "random",
        visualize: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
        # Render and access tensors
        self.env.isaac_gym.render_all_camera_sensors(self.env.sim)
        self.env.isaac_gym.start_access_image_tensors(self.env.sim)
        depth, seg = self._fetch_depth_and_seg(env_idx)
        depth = depth.clone()
        seg = seg.clone() if seg is not None else None
        self.env.isaac_gym.end_access_image_tensors(self.env.sim)

        # Visualize depth if debug enabled
        # if debug:
        #     self._visualize_depth(depth, env_idx)

        # isaac gym 会沿着 -z 轴给深度
        depth_flipped = False
        if float(depth.max().item()) <= 0.0:
            depth = -depth
            depth_flipped = True

        K = self._intrinsics()
        mask = self._crop(depth, seg, env_idx=env_idx)
        pts_sim_local = self._unproject(depth, K, mask, env_idx=env_idx)
        pts_frame = self._transform(pts_sim_local, env_idx=env_idx)
        pts_frame = self._downsample(pts_frame, max_points, mode=downsample_mode)
        if visualize:
            self._visualize_points(pts_frame)
        # debug 展示 crop 完的图片
        if debug and self._last_bbox_mask is not None:
            self._visualize_bbox_crop_on_image(env_idx, self._last_bbox_mask)
        return pts_frame

    def generate_transformed_cropped_point_cloud_for_all_env(
        self,
        max_points: Optional[int] = None,
        downsample_mode: str = "fps",
    ) -> List[torch.Tensor]:
        """Generate point clouds for all environments.

        Returns:
            List of (N, 3) tensors, one per environment.
        """
        if max_points is None:
            max_points = self.max_points

        num_envs = self.env.num_envs
        point_clouds = []
        for env_idx in range(num_envs):
            try:
                debug_show = env_idx == 0 and DEBUG_VISUALIZE
                pts = self.generate_transformed_cropped_point_cloud(
                    env_idx=env_idx,
                    max_points=max_points,
                    downsample_mode=downsample_mode,
                    visualize=debug_show,
                    debug=debug_show,
                )
                point_clouds.append(pts)
            except Exception as e:
                print(f"[PointCloudGenerator] Failed to generate PC for env {env_idx}: {e}")
                point_clouds.append(torch.zeros((max_points, 3), device=self.device))
        return point_clouds
