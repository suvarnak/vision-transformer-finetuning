import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
import numpy as np
from PIL import Image
import json
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# Import model classes
class DinoDetector(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Linear(768, num_classes)
        self.bbox_head = nn.Linear(768, 4)
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        
        logits = self.classifier(cls_token)
        bbox = self.bbox_head(cls_token)
        
        return {"logits": logits, "bbox": bbox}

class ImprovedDinoDetector(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.feature_proj = nn.Linear(768, 512)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        
        cls_token = outputs.last_hidden_state[:, 0]
        mean_pooled = outputs.last_hidden_state[:, 1:].mean(dim=1)
        
        combined_features = cls_token + mean_pooled
        features = self.feature_proj(combined_features)
        
        logits = self.classifier(features)
        bbox = self.bbox_head(features)
        
        return {"logits": logits, "bbox": bbox}

def preprocess_for_validation(example, processor):
    image = example["image"]
    annotations = example["objects"]
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].squeeze(0)
    
    if len(annotations["bbox"]) > 0:
        bbox = annotations["bbox"][0]
        label = annotations["category"][0]
        w, h = image.size
        bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
    else:
        bbox_norm = [0, 0, 0, 0]
        label = 0
    
    return {
        "pixel_values": pixel_values,
        "bbox": torch.tensor(bbox_norm, dtype=torch.float32),
        "label": torch.tensor(label, dtype=torch.long),
        "image": image,
        "all_bboxes": annotations["bbox"],
        "all_labels": annotations["category"]
    }

def calculate_iou(pred_bbox, true_bbox):
    """Calculate IoU between predicted and true bounding boxes"""
    pred_x1, pred_y1, pred_w, pred_h = pred_bbox
    true_x1, true_y1, true_w, true_h = true_bbox
    
    pred_x2 = pred_x1 + pred_w
    pred_y2 = pred_y1 + pred_h
    true_x2 = true_x1 + true_w
    true_y2 = true_y1 + true_h
    
    inter_x1 = max(pred_x1, true_x1)
    inter_y1 = max(pred_y1, true_y1)
    inter_x2 = min(pred_x2, true_x2)
    inter_y2 = min(pred_y2, true_y2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    pred_area = pred_w * pred_h
    true_area = true_w * true_h
    union_area = pred_area + true_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation"""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap

def validate_dino_model(model_path, model_class, test_dataset, processor, model_name):
    """Comprehensive validation for DINO models"""
    print(f"\nValidating {model_name}...")
    
    model = model_class()
    if model_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    elif model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    
    all_pred_scores = []
    all_pred_labels = []
    all_true_labels = []
    all_ious = []
    
    # For mAP calculation
    detections = []
    ground_truths = []
    
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            if i >= 500:  # Limit for faster validation
                break
                
            processed = preprocess_for_validation(example, processor)
            
            # Forward pass
            outputs = model(processed["pixel_values"].unsqueeze(0))
            
            pred_logits = outputs["logits"][0]
            pred_bbox = outputs["bbox"][0]
            
            # Get predictions
            pred_scores = torch.softmax(pred_logits, dim=0)
            pred_class = torch.argmax(pred_logits).item()
            pred_score = pred_scores[pred_class].item()
            
            true_class = processed["label"].item()
            
            # Store for metrics
            all_pred_scores.append(pred_score)
            all_pred_labels.append(pred_class)
            all_true_labels.append(true_class)
            
            # IoU calculation
            pred_bbox_np = pred_bbox.cpu().numpy()
            true_bbox_np = processed["bbox"].cpu().numpy()
            iou = calculate_iou(pred_bbox_np, true_bbox_np)
            all_ious.append(iou)
            
            # Store detections for mAP
            detections.append({
                'image_id': i,
                'category_id': pred_class,
                'bbox': pred_bbox_np.tolist(),
                'score': pred_score
            })
            
            ground_truths.append({
                'image_id': i,
                'category_id': true_class,
                'bbox': true_bbox_np.tolist()
            })
    
    # Calculate metrics
    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)
    all_pred_scores = np.array(all_pred_scores)
    
    # Classification accuracy
    accuracy = np.mean(all_pred_labels == all_true_labels)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, average='weighted', zero_division=0
    )
    
    # Mean IoU
    mean_iou = np.mean(all_ious)
    
    # Calculate mAP@0.5
    iou_threshold = 0.5
    correct_detections = np.array(all_ious) >= iou_threshold
    
    # Sort by confidence scores
    sorted_indices = np.argsort(all_pred_scores)[::-1]
    sorted_correct = correct_detections[sorted_indices]
    
    # Calculate precision and recall at each threshold
    tp = np.cumsum(sorted_correct)
    fp = np.cumsum(~sorted_correct)
    
    total_positives = len(all_true_labels)
    precisions = tp / (tp + fp + 1e-8)
    recalls = tp / (total_positives + 1e-8)
    
    # Calculate AP
    map_50 = calculate_ap(precisions, recalls)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'map_50': map_50,
        'total_samples': len(all_pred_labels)
    }

def validate_yolov8(test_dataset, model_size='n'):
    """Validate YOLOv8 model"""
    try:
        from ultralytics import YOLO
        
        print(f"\nValidating YOLOv8{model_size}...")
        model = YOLO(f'yolov8{model_size}.pt')
        
        all_pred_labels = []
        all_true_labels = []
        all_pred_scores = []
        all_ious = []
        
        for i, example in enumerate(test_dataset):
            if i >= 500:  # Limit for faster validation
                break
                
            image = example["image"]
            annotations = example["objects"]
            
            if len(annotations["bbox"]) == 0:
                continue
            
            # Run inference
            results = model(image, verbose=False)
            
            true_class = annotations["category"][0]
            true_bbox = annotations["bbox"][0]
            w, h = image.size
            true_bbox_norm = [true_bbox[0]/w, true_bbox[1]/h, true_bbox[2]/w, true_bbox[3]/h]
            
            if len(results[0].boxes) > 0:
                # Get best detection
                best_idx = torch.argmax(results[0].boxes.conf)
                pred_box = results[0].boxes[best_idx]
                pred_class = int(pred_box.cls.item())
                pred_score = pred_box.conf.item()
                pred_bbox = pred_box.xywhn[0].cpu().numpy()
                
                # IoU calculation
                iou = calculate_iou(pred_bbox, true_bbox_norm)
                all_ious.append(iou)
            else:
                pred_class = 0
                pred_score = 0.0
                all_ious.append(0.0)
            
            all_pred_labels.append(pred_class)
            all_true_labels.append(true_class)
            all_pred_scores.append(pred_score)
        
        # Calculate metrics
        all_pred_labels = np.array(all_pred_labels)
        all_true_labels = np.array(all_true_labels)
        all_pred_scores = np.array(all_pred_scores)
        
        accuracy = np.mean(all_pred_labels == all_true_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, all_pred_labels, average='weighted', zero_division=0
        )
        mean_iou = np.mean(all_ious)
        
        # Calculate mAP@0.5
        correct_detections = np.array(all_ious) >= 0.5
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        sorted_correct = correct_detections[sorted_indices]
        
        tp = np.cumsum(sorted_correct)
        fp = np.cumsum(~sorted_correct)
        total_positives = len(all_true_labels)
        precisions = tp / (tp + fp + 1e-8)
        recalls = tp / (total_positives + 1e-8)
        map_50 = calculate_ap(precisions, recalls)
        
        return {
            'model_name': f'YOLOv8{model_size}',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_iou': mean_iou,
            'map_50': map_50,
            'total_samples': len(all_pred_labels)
        }
        
    except ImportError:
        print("YOLOv8 validation skipped - ultralytics not installed")
        return None

def main():
    # Load test dataset
    print("Loading COCO validation dataset...")
    test_dataset = load_dataset("detection-datasets/coco", split="val[:500]")
    
    # Initialize processor
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    print("Starting comprehensive validation...")
    
    results = []
    
    # Validate Simple DINO
    simple_results = validate_dino_model(
        "dino2_detector.pth", 
        DinoDetector, 
        test_dataset, 
        processor, 
        "Simple DINO Detector"
    )
    results.append(simple_results)
    
    # Validate Improved DINO
    improved_results = validate_dino_model(
        "improved_dino2_detector.pth", 
        ImprovedDinoDetector, 
        test_dataset, 
        processor, 
        "Improved DINO Detector"
    )
    results.append(improved_results)
    
    # Validate YOLOv8n
    yolo_n_results = validate_yolov8(test_dataset, 'n')
    if yolo_n_results:
        results.append(yolo_n_results)
    
    # Validate YOLOv8x
    yolo_x_results = validate_yolov8(test_dataset, 'x')
    if yolo_x_results:
        results.append(yolo_x_results)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION RESULTS")
    print("="*80)
    
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Mean IoU':<10} {'mAP@0.5':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model_name']:<25} "
              f"{result['accuracy']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['f1_score']:<10.4f} "
              f"{result['mean_iou']:<10.4f} "
              f"{result['map_50']:<10.4f}")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  • Classification Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.1f}%)")
        print(f"  • Precision: {result['precision']:.4f}")
        print(f"  • Recall: {result['recall']:.4f}")
        print(f"  • F1-Score: {result['f1_score']:.4f}")
        print(f"  • Mean IoU: {result['mean_iou']:.4f}")
        print(f"  • mAP@0.5: {result['map_50']:.4f}")
        print(f"  • Total Samples: {result['total_samples']}")
    
    # Save results to JSON
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'validation_results.json'")

if __name__ == "__main__":
    main()