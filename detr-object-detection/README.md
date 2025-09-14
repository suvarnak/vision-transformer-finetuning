# DETR with DINOv3 Backbone

Meta's official DETR implementation with DINOv3 backbone for object detection on COCO dataset.

## Files

- `detr_model_meta.py` - DETR model with DINOv3 backbone
- `detr_loss.py` - Hungarian matching and loss functions  
- `train_detr_official.py` - Meta's official training procedure
- `transforms.py` - Meta's official data transforms
- `check_dinov3_dims.py` - Model dimension verification
- `requirements.txt` - Dependencies

## Installation

### 1. Install uv (Python package manager)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (run this or restart your shell)
source $HOME/.local/bin/env
```

### 2. Install dependencies and create environment
```bash
# Create virtual environment and install all dependencies
uv sync

# This automatically:
# - Creates .venv/ if it doesn't exist
# - Installs all dependencies from pyproject.toml
# - Includes dev dependencies
```

### 3. Activate environment
```bash
# Activate the created environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### 4. Login to HuggingFace (for DINOv3 access)
```bash
huggingface-cli login
# Enter your HF token when prompted
```

## Usage

### Training
```bash
python train_detr_official.py
```

Uses Meta's official DETR training procedure:
- Multi-scale transforms (480-800px)
- 300 epochs with lr scheduling
- Backbone unfreezing after 100 epochs
- Checkpoints saved every 50 epochs

## Model Architecture

- **DINOv3-vitl16 backbone** (1024 dimensions)
- Transformer encoder-decoder
- 100 object queries
- 256 hidden dimensions (DETR standard)
- Hungarian matching for loss computation
- Data augmentation (flip, color jitter, rotation)
- 50 training epochs on 5K COCO images

## Checking Model Dimensions

To verify DINOv2 and DINOv3 model dimensions:

```bash
# Check model dimensions
python check_dinov3_dims.py
```

### Expected Output:
```
=== DINOv2 Model Dimensions ===
facebook/dinov2-small: 384 dimensions
facebook/dinov2-base: 768 dimensions
facebook/dinov2-large: 1024 dimensions
facebook/dinov2-giant: 1536 dimensions

=== DINOv3 Model Dimensions ===
facebook/dinov3-vits16-pretrain-lvd1689m: 384 dimensions
facebook/dinov3-vitb16-pretrain-lvd1689m: 768 dimensions
facebook/dinov3-vitl16-pretrain-lvd1689m: 1024 dimensions
facebook/dinov3-vit7b16-pretrain-lvd1689m: 1536 dimensions
```

### Model Naming Convention:
- **DINOv2**: `dinov2-{small|base|large|giant}`
- **DINOv3**: `dinov3-{vits16|vitb16|vitl16|vit7b16}-pretrain-lvd1689m`

### Architecture Mapping:
| Model | DINOv2 | DINOv3 | Dimensions | Patch Size |
|-------|--------|--------|-----------|------------|
| Small | dinov2-small | dinov3-vits16 | 384 | 14x14 / 16x16 |
| Base | dinov2-base | dinov3-vitb16 | 768 | 14x14 / 16x16 |
| Large | dinov2-large | dinov3-vitl16 | 1024 | 14x14 / 16x16 |
| Giant | dinov2-giant | dinov3-vit7b16 | 1536 | 14x14 / 16x16 |

### Patch Size Details:
- **DINOv2**: Uses 14x14 patches
- **DINOv3**: Uses 16x16 patches (indicated by "16" in model name)
- **COCO Dataset**: Original images are variable size (32px-640px), typically ~480x640
- **Training Input Sizes** (after resize): 
  - **224x224**: DINOv2 = 16x16 patches, DINOv3 = 14x14 patches
  - **448x448**: DINOv2 = 32x32 patches, DINOv3 = 28x28 patches  
  - **512x512**: DINOv2 = 36x36 patches, DINOv3 = 32x32 patches
- **Meta's DETR**: Multi-scale 480-800px with max_size=1333px (aspect ratio preserving)

### Patch Grid Calculation:
```python
# For DINOv2 models
h_patches = input_height // 14
w_patches = input_width // 14

# For DINOv3 models  
h_patches = input_height // 16
w_patches = input_width // 16

# Meta's multi-scale examples:
# 480px input: DINOv3 = 30x30 patches
# 640px input: DINOv3 = 40x40 patches  
# 800px input: DINOv3 = 50x50 patches
```

## License

MIT License - see [LICENSE](LICENSE) file for details.