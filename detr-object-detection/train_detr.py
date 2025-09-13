import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import load_dataset
import os
from detr_model import DETR
from detr_loss import HungarianMatcher, SetCriterion
from transformers import AutoImageProcessor

class COCODataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process annotations
        annotations = item['objects']
        boxes = []
        labels = []
        
        img_w, img_h = 518, 518  # Fixed size after transform
        for i in range(len(annotations['bbox'])):
            bbox = annotations['bbox'][i]
            # Convert to cxcywh format and normalize
            x, y, w, h = bbox
            cx = (x + w/2) / img_w
            cy = (y + h/2) / img_h
            w = w / img_w
            h = h / img_h
            boxes.append([cx, cy, w, h])
            labels.append(annotations['category'][i])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
        
        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load COCO dataset
    print("Loading COCO dataset...")
    dataset = load_dataset("detection-datasets/coco", split="train[:5000]")
    
    # Data transforms with augmentation
    transform = T.Compose([
        T.Resize((224, 224)),  # Smaller input size
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    coco_dataset = COCODataset(dataset, transform)
    dataloader = DataLoader(coco_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)  # Reduce batch size
    
    # Initialize model with DINOv3
    model = DETR(num_classes=91, num_queries=50, hidden_dim=1024).to(device)
    
    # Initialize loss
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes']
    criterion = SetCriterion(91, matcher, weight_dict, eos_coef=0.1, losses=losses)
    criterion.to(device)
    
    # Optimizer with gradient accumulation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    accumulation_steps = 4  # Simulate batch_size=4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    num_epochs = 50  # More epochs for better convergence
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            losses_reduced = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses_reduced = losses_reduced / accumulation_steps  # Scale loss
            
            # Backward pass
            losses_reduced.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += losses_reduced.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses_reduced.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoints/detr_epoch_{epoch+1}.pth')
            print(f'Checkpoint saved: detr_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/detr_final.pth')
    print("Training completed. Final model saved as detr_final.pth")

if __name__ == "__main__":
    main()