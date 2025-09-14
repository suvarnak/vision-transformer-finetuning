import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from detr_model_meta import DETR
from detr_loss import HungarianMatcher, SetCriterion
from transforms import make_coco_transforms

class COCODataset:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        
        # Convert COCO format to DETR format
        annotations = item['objects']
        boxes = []
        labels = []
        
        for i in range(len(annotations['bbox'])):
            bbox = annotations['bbox'][i]
            x, y, w, h = bbox
            # Convert to xyxy format
            boxes.append([x, y, x + w, y + h])
            labels.append(annotations['category'][i])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor(idx),
            'area': torch.tensor([bbox[2] * bbox[3] for bbox in annotations['bbox']], dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'orig_size': torch.tensor([image.height, image.width]),
            'size': torch.tensor([image.height, image.width])
        }
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = (len(tensor_list),) + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return tensor, mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load COCO dataset
    print("Loading COCO dataset...")
    dataset = load_dataset("detection-datasets/coco", split="train[:5000]")
    
    # Meta's official transforms
    transforms = make_coco_transforms('train')
    
    # Create dataset and dataloader
    coco_dataset = COCODataset(dataset, transforms)
    dataloader = DataLoader(
        coco_dataset, 
        batch_size=2,  # Meta's original batch size per GPU
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model (Meta's config)
    model = DETR(
        num_classes=80,  # COCO has 80 detection classes
        num_queries=100,
        hidden_dim=256
    ).to(device)
    
    # Initialize loss (Meta's weights)
    matcher = HungarianMatcher(
        cost_class=1, 
        cost_bbox=5, 
        cost_giou=2
    )
    weight_dict = {
        'loss_ce': 1, 
        'loss_bbox': 5, 
        'loss_giou': 2
    }
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(
        80, 
        matcher, 
        weight_dict, 
        eos_coef=0.1, 
        losses=losses
    )
    criterion.to(device)
    
    # Meta's optimizer setup
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": 1e-5,  # Lower LR for backbone
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop (Meta's schedule)
    num_epochs = 300
    model.train()
    
    # Unfreeze backbone after 100 epochs
    unfreeze_epoch = 100
    
    for epoch in range(num_epochs):
        # Unfreeze backbone
        if epoch == unfreeze_epoch:
            print("Unfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
        
        total_loss = 0
        for batch_idx, (samples, targets) in enumerate(dataloader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(samples)
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            losses_reduced = sum(loss_dict[k] * weight_dict[k] 
                               for k in loss_dict.keys() if k in weight_dict)
            
            # Backward pass
            losses_reduced.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            total_loss += losses_reduced.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {losses_reduced.item():.4f}')
                # Print sample info
                if hasattr(samples, 'tensors'):
                    print(f'Sample shapes: {[s.shape for s in samples.tensors[:2]]}')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoints/detr_official_epoch_{epoch+1}.pth')
            print(f'Checkpoint saved: detr_official_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/detr_official_final.pth')
    print("Training completed. Final model saved as detr_official_final.pth")

if __name__ == "__main__":
    main()