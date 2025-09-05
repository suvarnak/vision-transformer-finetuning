# LoRA Fine-tuning for Vision Transformers

This folder contains code for parameter-efficient fine-tuning of Vision Transformers using LoRA (Low-Rank Adaptation).

## Overview

LoRA enables efficient fine-tuning of large Vision Transformers by adding trainable low-rank matrices to existing layers while keeping the original weights frozen.

## Dataset

- **CIFAR-10**: 10-class image classification
- **ImageNet**: 1000-class image classification
- Automatically downloaded via Hugging Face datasets

## Key Features

- **Parameter Efficiency**: Train only 1-5% of total parameters
- **Memory Reduction**: 50-70% less GPU memory usage
- **Performance**: Maintains 95%+ of full fine-tuning accuracy

## LoRA Implementation

```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=16):
        self.A = nn.Parameter(torch.randn(rank, original_layer.in_features))
        self.B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
    def forward(self, x):
        return original_layer(x) + (x @ self.A.T @ self.B.T)
```

## Usage

1. Open `lora.ipynb` in Jupyter notebook
2. Run all cells to:
   - Load pre-trained ViT model
   - Apply LoRA adaptation
   - Fine-tune on CIFAR-10/ImageNet
   - Evaluate performance

## Benefits

- **Faster Training**: Reduced computation requirements
- **Lower Memory**: Fits on smaller GPUs
- **Prevents Catastrophic Forgetting**: Original weights preserved
- **Easy Deployment**: Small adapter weights can be swapped

## Dependencies

```bash
pip install torch torchvision transformers datasets pillow accelerate
```

## Results

- Achieves comparable accuracy to full fine-tuning
- Uses significantly fewer trainable parameters
- Faster convergence and training time