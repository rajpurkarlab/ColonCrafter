import json
import os
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from src.submodules.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from src.submodules.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from typing import Optional, Tuple


class ColonCrafterInference(nn.Module):
    """
    Inference-only ColonCrafter for depth estimation.
    
    This class provides methods to predict depth maps from colonoscopy video sequences.
    """

    DEFAULT_CONFIG = {
        "unet_path": "tencent/DepthCrafter",
        "pretrained_path": "stabilityai/stable-video-diffusion-img2vid-xt",
        "lora_rank": 16,
        "lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
        "lora_dropout": 0.1,
        "chunk_size": 4,
        "fps": 7,
        "motion_bucket_id": 127,
        "noise_aug_strength": 0.0
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__()
        
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            self.config["unet_path"], 
            torch_dtype=torch.float16
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            self.config["pretrained_path"],
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        lora_config = LoraConfig(
            r=self.config["lora_rank"],
            lora_alpha=self.config["lora_rank"],
            target_modules=self.config["lora_target_modules"],
            lora_dropout=self.config["lora_dropout"],
            bias="none"
        )
        self.unet = get_peft_model(unet, lora_config)
        
        self.vae = self.pipe.vae
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        self.image_encoder = self.pipe.image_encoder
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

    @torch.inference_mode()
    def predict_depth(
        self,
        video: torch.Tensor,
        num_inference_steps: int = 1,
        window_size: int = 16,
        overlap: int = 8,
        guidance_scale: float = 1.0,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict depth from video frames.
        
        Args:
            video: Input video tensor of shape (N, C, H, W) in [0, 1] range.
            num_inference_steps: Number of denoising steps.
            window_size: Sliding window size for temporal processing.
            overlap: Overlap between consecutive windows.
            guidance_scale: Classifier-free guidance scale.
            
        Returns:
            Depth predictions of shape (N, H, W).
        """
        _, _, h, w = video.shape
        
        result = self.pipe(
            video,
            height=h,
            width=w,
            output_type="np",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            window_size=window_size,
            overlap=overlap,
            generator=torch.Generator(device=video.device).manual_seed(seed)
        )
        
        disparity = result.frames[0].mean(axis=-1)
        depth = 1.0 / disparity
        
        return depth, disparity

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "ColonCrafterInference":
        """
        Load a pretrained ColonCrafterInference model.
        
        Args:
            path: Local directory or HuggingFace Hub repo ID (e.g., "username/model-name").
            device: Device to load model on.
            
        Returns:
            Loaded model instance.
        """
        if os.path.isdir(path):
            config_path = os.path.join(path, "config.json")
            weights_path = os.path.join(path, "pytorch_model.bin")
        else:
            # Download from HuggingFace Hub
            try:
                config_path = hf_hub_download(repo_id=path, filename="config.json")
                weights_path = hf_hub_download(repo_id=path, filename="pytorch_model.bin")
            except Exception as e:
                raise ValueError(
                    f"Could not find model at local path '{path}' or on HuggingFace Hub: {e}"
                )
        
        config = None
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        
        model = cls(config)
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        
        return model.to(device)

    def save_pretrained(self, path: str) -> None:
        """
        Save model weights and config for HuggingFace.
        
        Args:
            path: Directory to save model to.
        """
        os.makedirs(path, exist_ok=True)
        
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        weights_path = os.path.join(path, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)