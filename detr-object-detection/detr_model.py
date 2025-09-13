import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as T
from transformers import pipeline

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
    def __init__(self, num_classes=91, num_queries=50, hidden_dim=1024):  # Reduce queries and dims
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # DINOv3 backbone
        self.backbone = pipeline(
            task="image-feature-extraction",
            model="facebook/dinov3-vits16-pretrain-lvd1689m",
            dtype=torch.bfloat16,
        )
        self.backbone_dim = 4096  # DINOv3 output dimension
        
        # Projection layer to match hidden_dim
        self.input_proj = nn.Conv2d(self.backbone_dim, hidden_dim, kernel_size=1)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer (reduced layers)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, 1024, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 3)  # Reduce layers
        
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, 8, 1024, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 3)  # Reduce layers
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x):
        # Extract features using DINOv3
        B = x.shape[0]
        
        # Convert tensor to PIL images for pipeline
        images = []
        for i in range(B):
            img_tensor = x[i]
            # Denormalize and convert to PIL
            img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img_tensor = torch.clamp(img_tensor, 0, 1)
            img_pil = T.ToPILImage()(img_tensor)
            images.append(img_pil)
        
        # Extract features
        with torch.no_grad():
            features_list = self.backbone(images)
        
        # Convert to tensor and reshape
        features_tensor = torch.stack([torch.tensor(f) for f in features_list])  # [B, num_patches, backbone_dim]
        
        # Get patch embeddings (excluding CLS token)
        patch_embeddings = features_tensor[:, 1:]  # [B, num_patches, backbone_dim]
        
        # Reshape to spatial format
        num_patches = patch_embeddings.shape[1]
        patch_size = int(num_patches ** 0.5)  # Assuming square patches
        features = patch_embeddings.transpose(1, 2).reshape(B, self.backbone_dim, patch_size, patch_size)
        
        # Project to hidden_dim
        features = self.input_proj(features)  # [B, hidden_dim, H, W]
        B, C, H, W = features.shape
        
        # Flatten and add positional encoding
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        features = self.pos_encoding(features)
        
        # Encoder
        memory = self.transformer_encoder(features)
        
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