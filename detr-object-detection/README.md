# DETR Object Detection

Classic DETR (DEtection TRansformer) implementation trained from scratch on COCO dataset.

## Files

- `detr_model.py` - Core DETR model with transformer architecture
- `detr_loss.py` - Hungarian matching and loss functions
- `train_detr.py` - Training script with checkpoint saving
- `validate_detr.py` - Validation script with metrics
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
python train_detr.py
```

Checkpoints saved every 5 epochs in `checkpoints/` directory.

### Validation
```bash
python validate_detr.py
```

Evaluates the trained model and reports precision, recall, and F1-score.

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
| Model | DINOv2 | DINOv3 | Dimensions |
|-------|--------|--------|-----------|
| Small | dinov2-small | dinov3-vits16 | 384 |
| Base | dinov2-base | dinov3-vitb16 | 768 |
| Large | dinov2-large | dinov3-vitl16 | 1024 |
| Giant | dinov2-giant | dinov3-vit7b16 | 1536 |

## License

MIT License - see [LICENSE](LICENSE) file for details.