# ColonCrafter

Depth estimation and style transfer for colonoscopy video sequences.

**[Paper](https://arxiv.org/abs/2509.13525)** | **[Model Weights](https://huggingface.co/romainhardy/coloncrafter)**

## Installation

Requires Python 3.10 and CUDA 11.8.

```bash
# Create an environment
conda create -n coloncrafter python=3.10 -y
conda activate coloncrafter

# Install PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

## Usage

### Depth Estimation

```python
import torch
from src.coloncrafter import ColonCrafterInference

# Load model
device = torch.device("cuda")
model = ColonCrafterInference.from_pretrained("romainhardy/coloncrafter", device=device)

# Predict depth from video frames (N, C, H, W) in [0, 1] range
depth, disparity = model.predict_depth(
    video,
    num_inference_steps=1,
    window_size=16,
    overlap=8,
)
```

### Style Transfer

```python
import torch
from src.style import StyleTransferPipeline2D

# Load pipeline
pipeline = StyleTransferPipeline2D(
    model_id="CompVis/stable-diffusion-v1-4",
    device="cuda",
    dtype=torch.float16,
)

# Transfer style from reference to content images
output = pipeline.run(
    images_content,  # (B, C, H, W) in [-1, 1]
    images_style,    # (B, C, H, W) in [-1, 1]
    num_inference_steps=25,
)
```

See `notebooks/` for complete examples.

## Citation

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={}
}
```

## License

This code is released for academic and research purposes only. Commercial use is prohibited due to dependencies on [DepthCrafter](https://github.com/Tencent/DepthCrafter) which has a non-commercial license.

The code is licensed under Apache 2.0. See [LICENSE](LICENSE) for details.