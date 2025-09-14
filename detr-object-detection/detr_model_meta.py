import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # DINOv3 backbone (using correct model identifier)
        self.backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
        self.backbone_dim = 1024  # DINOv3-vitl16 (large) output dimension
        
        # Freeze backbone initially (Meta's approach)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Projection layer to match hidden_dim
        self.input_proj = nn.Conv2d(self.backbone_dim, hidden_dim, kernel_size=1)
        
        # Positional encoding (DETR style)
        from position_encoding import PositionEmbeddingSine
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Transformer (Meta's full architecture)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, 2048, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 6)
        
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, 8, 2048, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 6)
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples):
        from nested_tensor import NestedTensor
        if isinstance(samples, NestedTensor):
            x, mask = samples.decompose()
        else:
            x = samples
            mask = None
        
        # Extract features using DINOv3
        B = x.shape[0]
        
        # Extract features with frozen backbone
        with torch.no_grad():
            dino_outputs = self.backbone(x)
        
        # Get patch embeddings (excluding CLS token)
        patch_embeddings = dino_outputs.last_hidden_state[:, 1:]  # [B, num_patches, backbone_dim]
        
        # Debug: print actual shapes
        print(f"Patch embeddings shape: {patch_embeddings.shape}")
        print(f"Input shape: {x.shape}")
        
        # Calculate patch grid from actual number of patches
        num_patches = patch_embeddings.shape[1]
        
        # Find the correct factorization
        factors = []
        for i in range(1, int(num_patches**0.5) + 1):
            if num_patches % i == 0:
                factors.append((i, num_patches // i))
        
        # Choose the most square-like factorization
        h_patches, w_patches = min(factors, key=lambda x: abs(x[0] - x[1]))
        
        print(f"Using patch grid: {h_patches} x {w_patches} = {h_patches * w_patches}")
        
        # Reshape to spatial format
        features = patch_embeddings.transpose(1, 2).reshape(B, self.backbone_dim, h_patches, w_patches)
        
        # Calculate target size for DETR (16x16 patches)
        target_h = x.shape[2] // 16
        target_w = x.shape[3] // 16
        
        # Interpolate to DETR expected size
        if h_patches != target_h or w_patches != target_w:
            features = torch.nn.functional.interpolate(
                features,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
        
        # Project to hidden_dim
        features = self.input_proj(features)  # [B, hidden_dim, H, W]
        B, C, H, W = features.shape
        
        # Create mask if not provided
        if mask is None:
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=features.device)
        else:
            # Resize mask to match features
            mask = torch.nn.functional.interpolate(
                mask.float().unsqueeze(1), 
                size=(H, W), 
                mode='nearest'
            ).squeeze(1).bool()
        
        # Add positional encoding (DETR style)
        from nested_tensor import NestedTensor
        nested_features = NestedTensor(features, mask)
        pos = self.position_embedding(nested_features)
        
        # Flatten for transformer
        src = features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        pos = pos.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        
        # Encoder (with positional encoding)
        memory = self.transformer_encoder(src + pos)
        
        # Decoder with object queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
        tgt = torch.zeros_like(query_embed)
        
        hs = self.transformer_decoder(tgt, memory)  # [num_queries, B, C]
        
        # Predictions
        outputs_class = self.class_embed(hs)  # [num_queries, B, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [num_queries, B, 4]
        
        return {
            'pred_logits': outputs_class.transpose(0, 1),  # [B, num_queries, num_classes+1]
            'pred_boxes': outputs_coord.transpose(0, 1)    # [B, num_queries, 4]
        }

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x