# DETR Object Detection

Classic DETR (DEtection TRansformer) implementation trained from scratch on COCO dataset.

## Files

- `detr_model.py` - Core DETR model with transformer architecture
- `detr_loss.py` - Hungarian matching and loss functions
- `train_detr.py` - Training script with checkpoint saving
- `validate_detr.py` - Validation script with metrics
- `requirements.txt` - Dependencies

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (run this or restart your shell)
source $HOME/.local/bin/env

# Install dependencies
uv pip install -r requirements.txt
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

- **DINOv3-vits16 backbone** (4096 dimensions)
- Transformer encoder-decoder
- 100 object queries
- 4096 hidden dimensions
- Hungarian matching for loss computation
- Data augmentation (flip, color jitter, rotation)
- 50 training epochs on 5K COCO images