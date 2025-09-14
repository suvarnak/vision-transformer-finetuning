# Architecture Comparison: Meta's DETR vs Our Implementation

## Meta's DETR with DINOv2 Backbone

```
Input Image (3, H, W)
        ↓
┌─────────────────────────┐
│   DINOv2-Large Backbone │
│   - Model: dinov2-large │
│   - Output: 1024 dim    │
│   - Patch size: 14x14   │
│   - Frozen initially    │
└─────────────────────────┘
        ↓ [B, 1024, 37, 37] (for 518x518 input)
┌─────────────────────────┐
│   Input Projection      │
│   - Conv2d(1024→256)    │
│   - Kernel: 1x1         │
└─────────────────────────┘
        ↓ [B, 256, 37, 37]
┌─────────────────────────┐
│   Positional Encoding  │
│   - Add 2D pos encoding │
└─────────────────────────┘
        ↓ [1369, B, 256] (flattened)
┌─────────────────────────┐
│   Transformer Encoder  │
│   - 6 layers            │
│   - 8 attention heads   │
│   - 256 hidden dim      │
│   - 2048 FFN dim        │
└─────────────────────────┘
        ↓ [1369, B, 256] (memory)
┌─────────────────────────┐
│   Object Queries        │
│   - 100 learnable       │
│   - 256 dimensions      │
└─────────────────────────┘
        ↓ [100, B, 256]
┌─────────────────────────┐
│   Transformer Decoder  │
│   - 6 layers            │
│   - 8 attention heads   │
│   - 256 hidden dim      │
│   - 2048 FFN dim        │
└─────────────────────────┘
        ↓ [100, B, 256]
┌─────────────────────────┐
│   Prediction Heads      │
│   - Class: Linear(256→92)│
│   - BBox: MLP(256→4)    │
└─────────────────────────┘
        ↓
Output: [B, 100, 92], [B, 100, 4]
```

## Our Implementation with DINOv3 Backbone

```
Input Image (3, H, W)
        ↓
┌─────────────────────────┐
│   DINOv3-Large Backbone │
│   - Model: dinov3-large │
│   - Output: 1024 dim    │
│   - Patch size: 14x14   │
│   - Frozen initially    │
└─────────────────────────┘
        ↓ [B, 1024, 37, 37] (for 518x518 input)
┌─────────────────────────┐
│   Input Projection      │
│   - Conv2d(1024→1024)   │ ← ISSUE: Should be 1024→256
│   - Kernel: 1x1         │
└─────────────────────────┘
        ↓ [B, 1024, 37, 37] ← ISSUE: Should be [B, 256, 37, 37]
┌─────────────────────────┐
│   Positional Encoding  │
│   - Add 2D pos encoding │
└─────────────────────────┘
        ↓ [1369, B, 1024] ← ISSUE: Should be [1369, B, 256]
┌─────────────────────────┐
│   Transformer Encoder  │
│   - 6 layers            │
│   - 8 attention heads   │
│   - 1024 hidden dim     │ ← ISSUE: Should be 256
│   - 2048 FFN dim        │
└─────────────────────────┘
        ↓ [1369, B, 1024] ← ISSUE: Should be [1369, B, 256]
┌─────────────────────────┐
│   Object Queries        │
│   - 100 learnable       │
│   - 1024 dimensions     │ ← ISSUE: Should be 256
└─────────────────────────┘
        ↓ [100, B, 1024] ← ISSUE: Should be [100, B, 256]
┌─────────────────────────┐
│   Transformer Decoder  │
│   - 6 layers            │
│   - 8 attention heads   │
│   - 1024 hidden dim     │ ← ISSUE: Should be 256
│   - 2048 FFN dim        │
└─────────────────────────┘
        ↓ [100, B, 1024] ← ISSUE: Should be [100, B, 256]
┌─────────────────────────┐
│   Prediction Heads      │
│   - Class: Linear(1024→92)← ISSUE: Should be Linear(256→92)
│   - BBox: MLP(1024→4)   │ ← ISSUE: Should be MLP(256→4)
└─────────────────────────┘
        ↓
Output: [B, 100, 92], [B, 100, 4]
```

## Key Issues Identified:

1. **Hidden Dimensions**: Our implementation uses 1024 throughout, but Meta's uses 256
2. **Input Projection**: Should project from backbone_dim (1024) to hidden_dim (256)
3. **Memory Usage**: Our approach uses 4x more memory than necessary
4. **Transformer Layers**: All layers should use 256 hidden dim, not 1024

## Corrected Architecture Should Be:

- **Backbone**: DINOv3-large (1024 dim output)
- **Projection**: 1024 → 256
- **Transformer**: 256 hidden dim throughout
- **Object Queries**: 256 dimensions
- **Prediction Heads**: Input 256 dimensions

This matches Meta's original DETR architecture while using DINOv3 backbone.