import copy
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from einops import rearrange
from joblib import Parallel, delayed
from kornia.color import rgb_to_lab
from kornia.morphology import dilation
from sklearn.linear_model import RANSACRegressor
from typing import Dict, List, Optional, Tuple, Union


MODELS = {
    "stabilityai/stable-diffusion-2-1": StableDiffusionPipeline,
    "stabilityai/stable-diffusion-2-1-base": StableDiffusionPipeline,
    "CompVis/stable-diffusion-v1-4": StableDiffusionPipeline,
}


class StyleTransferPipelineBase:
    """
    Base class for diffusion-based style transfer pipelines.
    
    Provides shared utilities for attention injection, latent inpainting,
    and histogram matching used by derived pipeline implementations.
    """

    DEFAULT_OPTIONS = {
        "gamma": 0.75,
        "tau": 1.5,
        "injection_layers": [7, 8, 9, 10, 11],
    }

    def __init__(
        self,
        model_id: str,
        options: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        for key, value in self.options.items():
            setattr(self, key, value)

    def _get_attention_layers(self, model) -> List:
        """Extract attention layers from UNet up blocks."""
        layers = []
        for i in range(12):
            up_block_idx = i // 3
            layer_idx = i % 3
            if up_block_idx > 0:
                layers.append(model.up_blocks[up_block_idx].attentions[layer_idx])
            else:
                layers.append(None)
        return layers

    def _reset_cache(self) -> None:
        """Initialize empty cache for attention features."""
        self.cache = {}
        for i in self.injection_layers:
            self.cache[f"layer{i}_attn"] = {}

    def _create_hooks(self) -> None:
        """Register forward hooks for attention injection."""
        for i in self.injection_layers:
            attn_module = self.attn_layers[i].transformer_blocks[0].attn1
            attn_module.register_forward_hook(self._get_qkv_hook_store(f"layer{i}_attn"))
            attn_module.register_forward_hook(self._get_qkv_hook_modify(f"layer{i}_attn"))

    def _get_qkv_hook_store(
        self, 
        name: str, 
        cache_device: torch.device = torch.device("cuda"),
        reduce_precision: bool = True
    ):
        """
        Create a hook that stores Q, K, V tensors during forward pass.
        
        Args:
            name: Cache key for storing attention features.
            cache_device: Device to store cached tensors on.
            reduce_precision: If True, store in float16 to save memory.
            
        Returns:
            Hook function for registering with attention module.
        """
        cache_dtype = torch.float16 if reduce_precision else self.dtype
        
        def hook(model, input, output):
            if self.trigger_store:
                _, q, k, v, _ = attention(model, input[0])
                self.cache[name][self.time_index] = tuple(
                    x.detach().to(cache_device, dtype=cache_dtype, non_blocking=True) 
                    for x in [q, k, v]
                )
        return hook

    def _get_qkv_hook_modify(self, name: str):
        """
        Create a hook that modifies attention using cached features.
        
        Args:
            name: Cache key for retrieving stored attention features.
            
        Returns:
            Hook function for registering with attention module.
        """
        def hook(model, input, output):
            if self.trigger_modify:
                _, q_cs, k_cs, v_cs, _ = attention(model, input[0])
                q_c, k_s, v_s = self.cache[name][self.time_index]
                q_c, k_s, v_s = [x.to(self.device, self.dtype, non_blocking=True) for x in [q_c, k_s, v_s]]
                
                q_hat_cs = q_c * self.gamma + q_cs * (1.0 - self.gamma)
                
                _, _, _, _, hidden_states = attention(
                    model, input[0], 
                    key=k_s, value=v_s, query=q_hat_cs, temperature=self.tau
                )
                return hidden_states
        return hook

    def _inpaint_latents(
        self,
        latents: torch.Tensor,
        patch_size: int = 16,
        overlap: int = 12,
        beta: float = 3.0,
        kernel_size: int = 3,
        n_jobs: int = -1,
    ) -> torch.Tensor:
        """
        Remove specular highlights from latents using patch-based inpainting.
        
        Args:
            latents: Input latent tensor of shape (B, C, H, W) or (B, F, C, H, W).
            patch_size: Size of patches for processing.
            overlap: Overlap between adjacent patches.
            beta: Threshold multiplier for specularity detection.
            kernel_size: Dilation kernel size for mask expansion.
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            
        Returns:
            Inpainted latent tensor with same shape as input.
        """
        latents, batch_size, was_5d = _flatten_video_batch(latents)
        step = patch_size - overlap
        
        window = torch.from_numpy(create_hann_window(patch_size)).to(latents.device, torch.float32)
        latents_padded = F.pad(latents, (step, step, step, step), "reflect")
        
        patches, positions = extract_patches(latents_padded.float(), patch_size, overlap)
        processed_patches = Parallel(n_jobs=n_jobs)(
            delayed(inpaint_patch)(patch, beta=beta, kernel_size=kernel_size) 
            for patch in patches
        )
        merged = merge_patches(processed_patches, positions, latents_padded.shape, patch_size, window)
        merged = merged[..., step:-step, step:-step].to(latents.device, latents.dtype)
        
        return _unflatten_video_batch(merged, batch_size, was_5d)

    def _match_histograms(
        self,
        output: np.ndarray,
        target: np.ndarray,
        patch_size: int = 128,
        overlap: int = 96,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """
        Match output histograms to target using patch-based RANSAC regression.
        
        Args:
            output: Output image array of shape (B, C, H, W) or (B, F, C, H, W).
            target: Target image array with same shape as output.
            patch_size: Size of patches for processing.
            overlap: Overlap between adjacent patches.
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            
        Returns:
            Histogram-matched output array with same shape as input.
        """
        output, batch_size, was_5d = _flatten_video_batch(output)
        target, _, _ = _flatten_video_batch(target)
        step = patch_size - overlap
        
        window = create_hann_window(patch_size)[None, ...]
        output_padded = np.pad(output, ((0, 0), (0, 0), (step, step), (step, step)), mode="reflect")
        target_padded = np.pad(target, ((0, 0), (0, 0), (step, step), (step, step)), mode="reflect")
        
        for i in range(output_padded.shape[0]):
            src_patches, positions = extract_patches(output_padded[i], patch_size, overlap)
            ref_patches, _ = extract_patches(target_padded[i], patch_size, overlap)
            matched_patches = Parallel(n_jobs=n_jobs)(
                delayed(match_patch)(sp, rp, patch_size) 
                for sp, rp in zip(src_patches, ref_patches)
            )
            output_padded[i] = merge_patches(
                matched_patches, positions, output_padded[i].shape, patch_size, window
            )
        
        result = output_padded[..., step:-step, step:-step]
        return _unflatten_video_batch(result, batch_size, was_5d)


class StyleTransferPipeline2D(StyleTransferPipelineBase):
    """
    2D style transfer pipeline using Stable Diffusion with attention injection.
    
    Transfers style from a reference image to content images using DDIM inversion
    and attention feature injection during the reverse diffusion process.
    """

    def __init__(
        self,
        model_id: str,
        options: Optional[Dict] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__(model_id, options, device, dtype)

        self._load_model(model_id)
        self.attn_layers = self._get_attention_layers(self.unet)

        self.trigger_store = False
        self.trigger_modify = False
        self._reset_cache()
        self._create_hooks()
        self.time_index = None

    def _load_model(self, model_id: str) -> None:
        """Load and initialize Stable Diffusion components."""
        if model_id not in MODELS:
            raise ValueError(f"Model {model_id} not supported. Available: {list(MODELS.keys())}")

        pipe = MODELS[model_id].from_pretrained(model_id, torch_dtype=self.dtype)

        self.vae = pipe.vae.eval().to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval().to(self.device)
        self.unet = pipe.unet.eval().to(self.device)
        self.unet.enable_xformers_memory_efficient_attention()
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def encode_vae(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.
        
        Args:
            images: Input images of shape (B, C, H, W) in [-1, 1] range.
            
        Returns:
            Latent tensor of shape (B, 4, H//8, W//8).
        """
        latents = self.vae.encode(images).latent_dist.mode()
        return self.vae.config.scaling_factor * latents

    @torch.inference_mode()
    def decode_vae(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to image space.
        
        Args:
            latents: Latent tensor of shape (B, 4, H//8, W//8).
            
        Returns:
            Decoded images of shape (B, C, H, W) in [-1, 1] range.
        """
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents).sample

    @torch.inference_mode()
    def get_text_conditioning(self) -> torch.Tensor:
        """Get unconditional text embeddings for classifier-free guidance."""
        text_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        return self.text_encoder(text_input.input_ids.to(self.device))[0]

    @torch.inference_mode()
    def invert_process(
        self, 
        latents: torch.Tensor, 
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform DDIM inversion to map latents to noise space.
        
        Args:
            latents: Input latent tensor.
            encoder_hidden_states: Text conditioning embeddings.
            
        Returns:
            Inverted latent tensor in noise space.
        """
        timesteps = list(reversed(self.scheduler.timesteps))
        num_steps = len(timesteps)

        for i, t in enumerate(timesteps[:self.inversion_stop_index]):
            self.time_index = t.item()
            noise_pred = self.unet(
                latents, t.to(self.device), 
                encoder_hidden_states=encoder_hidden_states, 
                return_dict=False
            )[0]

            t_next = timesteps[i + 1] if i < num_steps - 1 else 999
            alpha_t = self.scheduler.alphas_cumprod[t]
            alpha_t_next = self.scheduler.alphas_cumprod[t_next]

            if self.model_id == "stabilityai/stable-diffusion-2-1":
                beta_t = 1.0 - alpha_t
                pred_original = alpha_t.sqrt() * latents - beta_t.sqrt() * noise_pred
                pred_epsilon = alpha_t.sqrt() * noise_pred + beta_t.sqrt() * latents
                pred_direction = (1.0 - alpha_t_next).sqrt() * pred_epsilon
                latents = alpha_t_next.sqrt() * pred_original + pred_direction
            else:
                latents = (
                    (latents - (1.0 - alpha_t).sqrt() * noise_pred) 
                    * (alpha_t_next.sqrt() / alpha_t.sqrt()) 
                    + (1.0 - alpha_t_next).sqrt() * noise_pred
                )

        return latents

    @torch.inference_mode()
    def reverse_process(
        self, 
        latents: torch.Tensor, 
        encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform reverse diffusion with style-injected attention.
        
        Args:
            latents: Noisy latent tensor.
            encoder_hidden_states: Text conditioning embeddings.
            
        Returns:
            Denoised latent tensor.
        """
        for t in self.scheduler.timesteps[-self.inversion_stop_index:]:
            self.time_index = t.item()
            noise_pred = self.unet(
                latents, t.to(self.device),
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    @torch.inference_mode()
    def run(
        self,
        images_content: torch.Tensor,
        images_style: torch.Tensor,
        num_inference_steps: int = 20,
        partial_inversion_fraction: float = 1.0,
        alpha: float = 1.0,
        inpaint_latents: bool = False,
        match_histograms: bool = False,
    ) -> np.ndarray:
        """
        Transfer style from style images to content images.
        
        Args:
            images_content: Content images of shape (B, C, H, W) in [-1, 1] range.
            images_style: Style images of shape (B, C, H, W) in [-1, 1] range.
            num_inference_steps: Number of diffusion steps.
            partial_inversion_fraction: Fraction of steps for partial inversion.
            alpha: AdaIN blending factor (0=full style, 1=preserve content statistics).
            inpaint_latents: Whether to remove specular highlights from content.
            match_histograms: Whether to match output histograms to content.
            
        Returns:
            Stylized images of shape (B, H, W, C) in [0, 1] range.
        """
        batch_size = images_content.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        self.inversion_stop_index = int(partial_inversion_fraction * num_inference_steps)

        encoder_hidden_states = self.get_text_conditioning().repeat(batch_size, 1, 1)

        # Invert style image and cache attention features
        self.trigger_store = True
        self.trigger_modify = False

        latents_style = self.encode_vae(images_style.to(self.device, self.dtype))
        latents_style = self.invert_process(latents_style, encoder_hidden_states)
        feats_style = copy.deepcopy(self.cache)

        # Invert content image and cache attention features
        latents_content = self.encode_vae(images_content.to(self.device, self.dtype))
        if inpaint_latents:
            latents_content = self._inpaint_latents(latents_content)
        latents_content = self.invert_process(latents_content, encoder_hidden_states)
        feats_content = copy.deepcopy(self.cache)

        # Prepare injection cache: content queries + style keys/values
        for layer_name in feats_style.keys():
            self.cache[layer_name] = {}
            for t_ in self.scheduler.timesteps[-self.inversion_stop_index:]:
                t = t_.item()
                self.cache[layer_name][t] = (
                    feats_content[layer_name][t][0],  # content query
                    feats_style[layer_name][t][1],    # style key
                    feats_style[layer_name][t][2],    # style value
                )

        # Run reverse diffusion with style injection
        self.trigger_store = False
        self.trigger_modify = True

        latents = adain(latents_content, latents_style, alpha=alpha, dim=[2, 3])
        latents = latents.to(self.device, self.dtype)
        latents = self.reverse_process(latents, encoder_hidden_states)

        # Decode and postprocess
        images = self.decode_vae(latents)
        images = torch.clamp(images * 0.5 + 0.5, 0.0, 1.0)
        images = images.cpu().float().numpy()

        if match_histograms:
            target = images_content.cpu().float().numpy() * 0.5 + 0.5
            images = self._match_histograms(images, target)

        # Cleanup
        for layer_name in feats_style.keys():
            self.cache[layer_name] = {}
        torch.cuda.empty_cache()

        return np.transpose(images, (0, 2, 3, 1))


def _flatten_video_batch(
    tensor: Union[torch.Tensor, np.ndarray]
) -> Tuple[Union[torch.Tensor, np.ndarray], int, bool]:
    """
    Flatten 5D video tensor to 4D batch tensor.
    
    Args:
        tensor: Input of shape (B, C, H, W) or (B, F, C, H, W).
        
    Returns:
        Tuple of (flattened tensor, original batch size, was_5d flag).
    """
    batch_size = tensor.shape[0]
    was_5d = tensor.ndim == 5
    if was_5d:
        tensor = rearrange(tensor, "b f c h w -> (b f) c h w")
    return tensor, batch_size, was_5d


def _unflatten_video_batch(
    tensor: Union[torch.Tensor, np.ndarray], 
    batch_size: int, 
    was_5d: bool
) -> Union[torch.Tensor, np.ndarray]:
    """
    Restore 5D video tensor from flattened 4D batch.
    
    Args:
        tensor: Flattened tensor of shape (B*F, C, H, W).
        batch_size: Original batch dimension.
        was_5d: Whether to restore to 5D shape.
        
    Returns:
        Tensor of shape (B, C, H, W) or (B, F, C, H, W).
    """
    if was_5d:
        tensor = rearrange(tensor, "(b f) c h w -> b f c h w", b=batch_size)
    return tensor


def adain(
    x: torch.Tensor, 
    y: torch.Tensor, 
    dim: List[int] = [2, 3], 
    alpha: float = 0.0
) -> torch.Tensor:
    """
    Adaptive Instance Normalization for style transfer.
    
    Args:
        x: Content tensor.
        y: Style tensor.
        dim: Dimensions to compute statistics over.
        alpha: Blending factor (0=full style, 1=preserve content statistics).
        
    Returns:
        Normalized tensor with blended statistics.
    """
    mu_x = x.mean(dim=dim, keepdim=True)
    sigma_x = x.std(dim=dim, keepdim=True)
    mu_y = y.mean(dim=dim, keepdim=True)
    sigma_y = y.std(dim=dim, keepdim=True)
    
    mu = alpha * mu_x + (1.0 - alpha) * mu_y
    sigma = alpha * sigma_x + (1.0 - alpha) * sigma_y
    
    return mu + sigma * (x - mu_x) / (sigma_x + 1e-5)


def attention(
    module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    attention_probs: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute attention with optional Q/K/V override for style injection.
    
    Args:
        module: Attention module from diffusers.
        hidden_states: Input hidden states.
        encoder_hidden_states: Cross-attention conditioning (optional).
        attention_mask: Attention mask (optional).
        query: Override query tensor (optional).
        key: Override key tensor (optional).
        value: Override value tensor (optional).
        attention_probs: Override attention probabilities (optional).
        temperature: Temperature scaling for attention logits.
        
    Returns:
        Tuple of (attention_probs, query, key, value, output_hidden_states).
    """
    residual = hidden_states
    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        b, c, h, w = hidden_states.shape
        hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

    b, s, _ = (
        hidden_states.shape if encoder_hidden_states is None 
        else encoder_hidden_states.shape
    )
    attention_mask = module.prepare_attention_mask(attention_mask, s, b)

    if module.group_norm is not None:
        hidden_states = module.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    if query is None:
        query = module.head_to_batch_dim(module.to_q(hidden_states))

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif module.norm_cross:
        encoder_hidden_states = module.norm_encoder_hidden_states(encoder_hidden_states)

    if key is None:
        key = module.head_to_batch_dim(module.to_k(encoder_hidden_states))
    if value is None:
        value = module.head_to_batch_dim(module.to_v(encoder_hidden_states))

    if key.shape[0] != query.shape[0]:
        key = key[:query.shape[0]]
        value = value[:query.shape[0]]

    query = query * temperature

    if attention_probs is None:
        attention_probs = module.get_attention_scores(query, key, attention_mask)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = module.batch_to_head_dim(hidden_states)
    hidden_states = module.to_out[0](hidden_states)
    hidden_states = module.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

    if module.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / module.rescale_output_factor

    return attention_probs, query, key, value, hidden_states


def create_hann_window(patch_size: int, eps: float = 1e-10) -> np.ndarray:
    """Create 2D Hann window for smooth patch blending."""
    one_d = np.hanning(patch_size)
    return np.outer(one_d, one_d) + eps


def extract_patches(
    image: Union[torch.Tensor, np.ndarray], 
    patch_size: int, 
    overlap: int
) -> Tuple[List, List[Tuple[int, int]]]:
    """
    Extract overlapping patches from an image.
    
    Args:
        image: Input image of shape (..., H, W).
        patch_size: Size of square patches.
        overlap: Overlap between adjacent patches.
        
    Returns:
        Tuple of (list of patches, list of (row, col) positions).
    """
    step = patch_size - overlap
    h, w = image.shape[-2:]
    patches = []
    positions = []
    
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            patches.append(image[..., i:i + patch_size, j:j + patch_size])
            positions.append((i, j))
    
    return patches, positions


def merge_patches(
    patches: List,
    positions: List[Tuple[int, int]],
    shape: Tuple,
    patch_size: int,
    window: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Merge overlapping patches back into a single image.
    
    Args:
        patches: List of patch tensors/arrays.
        positions: List of (row, col) positions for each patch.
        shape: Output shape.
        patch_size: Size of square patches.
        window: Blending window weights.
        
    Returns:
        Merged image tensor/array.
    """
    if isinstance(patches[0], torch.Tensor):
        merged = torch.zeros(shape, device=patches[0].device, dtype=patches[0].dtype)
        weight_sum = torch.zeros(shape, device=patches[0].device, dtype=patches[0].dtype)
    else:
        merged = np.zeros(shape)
        weight_sum = np.zeros(shape)

    for patch, (i, j) in zip(patches, positions):
        merged[..., i:i + patch_size, j:j + patch_size] += patch * window
        weight_sum[..., i:i + patch_size, j:j + patch_size] += window

    return merged / weight_sum


def get_specularity_mask(patch: torch.Tensor, beta: float = 3.0) -> torch.Tensor:
    """
    Detect specular highlights using statistical thresholding.
    
    Args:
        patch: Input patch of shape (B, C, H, W).
        beta: Threshold multiplier (higher = fewer detections).
        
    Returns:
        Binary mask of shape (B, 1, H, W).
    """
    if patch.shape[1] == 3:
        x = rgb_to_lab(patch)[:, 0:1, :, :]
    else:
        x = patch.max(dim=1, keepdim=True).values
    
    mu = x.mean(dim=(2, 3), keepdim=True)
    sigma = x.std(dim=(2, 3), keepdim=True)
    threshold = mu + beta * sigma
    
    return (x > threshold).float()


def inpaint_patch(
    patch: torch.Tensor, 
    beta: float = 3.0, 
    kernel_size: int = 3
) -> torch.Tensor:
    """
    Inpaint specular highlights in a patch using median replacement.
    
    Args:
        patch: Input patch of shape (B, C, H, W).
        beta: Threshold multiplier for specularity detection.
        kernel_size: Dilation kernel size for mask expansion.
        
    Returns:
        Inpainted patch with same shape.
    """
    mask = get_specularity_mask(patch, beta)
    kernel = torch.ones(kernel_size, kernel_size, device=patch.device, dtype=patch.dtype)
    mask = dilation(mask, kernel).bool()
    
    median = torch.median(
        rearrange(patch, "b c h w -> b c (h w)"), dim=-1, keepdim=True
    ).values.unsqueeze(-1).expand_as(patch)
    
    return torch.where(mask, median, patch)


def match_patch(
    src_patch: np.ndarray,
    ref_patch: np.ndarray,
    patch_size: int,
    max_trials: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Match source patch histogram to reference using RANSAC regression.
    
    Args:
        src_patch: Source patch of shape (C, H, W).
        ref_patch: Reference patch of shape (C, H, W).
        patch_size: Patch dimension size.
        max_trials: Maximum RANSAC iterations.
        random_state: Random seed for reproducibility.
        
    Returns:
        Matched patch of shape (C, H, W).
    """
    t = ref_patch.mean(axis=0).reshape(-1, 1)
    p = src_patch.mean(axis=0).reshape(-1, 1)
    
    model = RANSACRegressor(random_state=random_state, max_trials=max_trials)
    model.fit(p, t.ravel())
    
    pred = model.predict(p).reshape((patch_size, patch_size))
    pred = np.clip(pred, 0.0, 1.0)
    
    return np.repeat(pred[None, ...], 3, axis=0)
