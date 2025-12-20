import cv2
import glob
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class C3VDDataset(Dataset):
    """
    Dataset loader for C3VD colonoscopy video sequences.
    
    Loads RGB frames, depth maps, and camera poses from the C3VD dataset format.
    """

    def __init__(
        self,
        root_dir: str,
        section: str,
        resize: Optional[Tuple[int, int]] = (512, 512),
        depth_scale: float = 2.55,
        use_depth: bool = True,
    ) -> None:
        """
        Initialize the C3VD dataset.
        
        Args:
            root_dir: Root directory containing C3VD data.
            section: Section name (subdirectory) to load.
            resize: Target (width, height) for resizing, or None to keep original.
            depth_scale: Scale factor to convert depth values to metric units.
            use_depth: Whether to load depth maps.
        """
        self.root_dir = root_dir
        self.section = section
        self.resize = resize
        self.depth_scale = depth_scale
        self.use_depth = use_depth

        self.image_paths = self._load_paths(key="color", file_type="png")
        if self.use_depth:
            self.depth_paths = self._load_paths(key="depth", file_type="tiff")
        self.poses = self._load_poses()

    def _load_paths(self, key: str, file_type: str) -> List[str]:
        """Load sorted file paths for a given data key."""
        directory = os.path.join(self.root_dir, self.section, key)
        return sorted(glob.glob(os.path.join(directory, f"*.{file_type}")))

    def _load_poses(self) -> List[np.ndarray]:
        """Load camera poses from pose.txt file."""
        path = os.path.join(self.root_dir, self.section, "pose.txt")
        poses = []

        if not os.path.exists(path):
            return poses

        with open(path, "r") as f:
            for line in f:
                values = list(map(float, line.strip().split(",")))
                if len(values) != 16:
                    continue

                if _is_column_major(values):
                    pose = np.array(values, dtype=np.float32).reshape(4, 4, order="C")
                else:
                    pose = np.array(values, dtype=np.float32).reshape(4, 4, order="F")

                poses.append(pose)

        return poses

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an RGB image."""
        image = np.array(Image.open(path)) / 255.0
        if self.resize is not None:
            image = cv2.resize(image, self.resize, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(image).permute(2, 0, 1)

    def _load_depth(self, path: str) -> torch.Tensor:
        """Load and preprocess a depth map."""
        depth = np.array(Image.open(path))
        if self.resize is not None:
            depth = cv2.resize(depth, self.resize, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(depth).float() / self.depth_scale

    def _load_pose(self, idx: int) -> torch.Tensor:
        """Load camera pose for a given frame index."""
        if len(self.poses) > 0:
            return torch.from_numpy(self.poses[idx])
        return torch.eye(4, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing:
                - "image": RGB tensor of shape (3, H, W) in [-1, 1] range.
                - "pose": Camera pose matrix of shape (4, 4).
                - "depth": Depth tensor of shape (H, W) in metric units (if use_depth=True).
        """
        result = {
            "image": self._load_image(self.image_paths[idx]) * 2.0 - 1.0,
            "pose": self._load_pose(idx),
        }
        if self.use_depth:
            result["depth"] = self._load_depth(self.depth_paths[idx])
        return result


def _is_column_major(values: List[float], eps: float = 1e-6) -> bool:
    """
    Detect if pose values are stored in column-major order.
    
    Args:
        values: Flat list of 16 pose matrix values.
        eps: Tolerance for floating point comparison.
        
    Returns:
        True if values appear to be column-major ordered.
    """
    return (
        abs(values[12]) < eps
        and abs(values[13]) < eps
        and abs(values[14]) < eps
        and abs(values[15] - 1.0) < eps
        and (abs(values[3]) > eps or abs(values[7]) > eps or abs(values[11]) > eps)
    )