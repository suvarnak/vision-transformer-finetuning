import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
import torch.nn.functional as F

class ImprovedDinoDetector(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Multi-scale feature extraction
        self.feature_proj = nn.Linear(768, 512)
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Bbox regression head with multiple layers
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Ensure bbox values are in [0,1]
        )
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        
        # Use both CLS token and mean pooling
        cls_token = outputs.last_hidden_state[:, 0]
        mean_pooled = outputs.last_hidden_state[:, 1:].mean(dim=1)
        
        # Combine features
        combined_features = cls_token + mean_pooled
        features = self.feature_proj(combined_features)
        
        logits = self.classifier(features)
        bbox = self.bbox_head(features)
        
        return {"logits": logits, "bbox": bbox}

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss for bbox regression"""
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return loss.mean()

def improved_compute_loss(outputs, targets):
    logits = outputs["logits"]
    bbox_pred = outputs["bbox"]
    
    labels = targets["labels"]
    bbox_true = targets["bboxes"]
    
    # Use focal loss for classification
    cls_loss = focal_loss(logits, labels)
    
    # Use smooth L1 loss for bbox regression
    bbox_loss = smooth_l1_loss(bbox_pred, bbox_true)
    
    # Weighted combination
    total_loss = cls_loss + 2.0 * bbox_loss
    
    return total_loss

def improved_training_loop():
    # Load dataset
    dataset = load_dataset("detection-datasets/coco", split="train[:5000]")  # More data
    
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = ImprovedDinoDetector()
    
    # Data preprocessing with collate function
    from finetune_dino2_simple import preprocess_coco, collate_fn
    
    processed_data = []
    for i in range(len(dataset)):
        example = dataset[i]
        processed_example = preprocess_coco(example, processor)
        processed_data.append(processed_example)
    
    from torch.utils.data import Dataset as TorchDataset
    
    class SimpleDataset(TorchDataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    processed_dataset = SimpleDataset(processed_data)
    
    # Improved training setup
    dataloader = DataLoader(processed_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # Two-stage training
    # Stage 1: Train only heads
    optimizer = torch.optim.AdamW([
        {'params': model.feature_proj.parameters()},
        {'params': model.classifier.parameters()},
        {'params': model.bbox_head.parameters()}
    ], lr=1e-3, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print("Stage 1: Training heads only...")
    model.train()
    
    for epoch in range(10):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(batch["pixel_values"])
            loss = improved_compute_loss(outputs, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        print(f"Epoch {epoch} completed. Average loss: {total_loss/len(dataloader):.4f}")
    
    # Stage 2: Fine-tune entire model
    print("Stage 2: Fine-tuning entire model...")
    
    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    # Lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    for epoch in range(5):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(batch["pixel_values"])
            loss = improved_compute_loss(outputs, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Fine-tune Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        print(f"Fine-tune Epoch {epoch} completed. Average loss: {total_loss/len(dataloader):.4f}")
    
    # Save improved model
    torch.save(model.state_dict(), "improved_dino2_detector.pth")
    print("Improved model saved!")

if __name__ == "__main__":
    # You'll need to import preprocess_coco from the original script
    from finetune_dino2_simple import preprocess_coco
    improved_training_loop()