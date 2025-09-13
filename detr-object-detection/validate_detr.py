import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import load_dataset
import numpy as np
from detr_model import DETR
from detr_loss import box_cxcywh_to_xyxy, box_iou

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

@torch.no_grad()
def evaluate(model, dataloader, device, confidence_threshold=0.5):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        
        # Process predictions
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Apply softmax to get probabilities
        pred_probs = F.softmax(pred_logits, dim=-1)
        
        for i in range(len(images)):
            # Get predictions for this image
            probs = pred_probs[i]
            boxes = pred_boxes[i]
            
            # Filter by confidence threshold
            max_probs, pred_classes = probs[:, :-1].max(dim=-1)  # Exclude no-object class
            keep = max_probs > confidence_threshold
            
            if keep.sum() > 0:
                pred_dict = {
                    'boxes': boxes[keep],
                    'labels': pred_classes[keep],
                    'scores': max_probs[keep]
                }
            else:
                pred_dict = {
                    'boxes': torch.empty((0, 4), device=device),
                    'labels': torch.empty((0,), dtype=torch.long, device=device),
                    'scores': torch.empty((0,), device=device)
                }
            
            all_predictions.append(pred_dict)
            all_targets.append(targets[i])
    
    return all_predictions, all_targets

def compute_ap(predictions, targets, iou_threshold=0.5):
    """Compute Average Precision for a single IoU threshold"""
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    
    for pred, target in zip(predictions, targets):
        if len(pred['boxes']) > 0:
            all_pred_boxes.append(box_cxcywh_to_xyxy(pred['boxes']))
            all_pred_scores.append(pred['scores'])
            all_pred_labels.append(pred['labels'])
        else:
            all_pred_boxes.append(torch.empty((0, 4)))
            all_pred_scores.append(torch.empty((0,)))
            all_pred_labels.append(torch.empty((0,), dtype=torch.long))
        
        if len(target['boxes']) > 0:
            all_gt_boxes.append(box_cxcywh_to_xyxy(target['boxes']))
            all_gt_labels.append(target['labels'])
        else:
            all_gt_boxes.append(torch.empty((0, 4)))
            all_gt_labels.append(torch.empty((0,), dtype=torch.long))
    
    # Simple AP calculation (simplified version)
    total_tp = 0
    total_fp = 0
    total_gt = sum(len(gt) for gt in all_gt_labels)
    
    for pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels in zip(
        all_pred_boxes, all_pred_scores, all_pred_labels, all_gt_boxes, all_gt_labels
    ):
        if len(pred_boxes) == 0:
            continue
            
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        # Compute IoU between all predictions and ground truth
        ious, _ = box_iou(pred_boxes, gt_boxes)
        
        # For each prediction, find best matching ground truth
        for i in range(len(pred_boxes)):
            if len(gt_boxes) > 0:
                best_iou = ious[i].max()
                if best_iou > iou_threshold:
                    total_tp += 1
                else:
                    total_fp += 1
            else:
                total_fp += 1
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    
    return precision, recall

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load validation dataset
    print("Loading COCO validation dataset...")
    dataset = load_dataset("detection-datasets/coco", split="validation[:1000]")
    
    # Data transforms
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    coco_dataset = COCODataset(dataset, transform)
    dataloader = DataLoader(coco_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = DETR(num_classes=91, num_queries=100, hidden_dim=4096).to(device)
    
    # Load trained model
    checkpoint_path = 'checkpoints/detr_final.pth'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    print("Model loaded successfully")
    
    # Evaluate model
    print("Starting evaluation...")
    predictions, targets = evaluate(model, dataloader, device, confidence_threshold=0.5)
    
    # Compute metrics
    precision, recall = compute_ap(predictions, targets, iou_threshold=0.5)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nValidation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Count predictions and targets
    total_predictions = sum(len(pred['boxes']) for pred in predictions)
    total_targets = sum(len(target['boxes']) for target in targets)
    
    print(f"\nStatistics:")
    print(f"Total predictions: {total_predictions}")
    print(f"Total ground truth objects: {total_targets}")
    print(f"Average predictions per image: {total_predictions / len(predictions):.2f}")
    print(f"Average ground truth per image: {total_targets / len(targets):.2f}")

if __name__ == "__main__":
    main()