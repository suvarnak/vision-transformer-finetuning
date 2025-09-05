import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2

# Import DINO detector from training script
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

def preprocess_for_validation(example, processor):
    image = example["image"]
    annotations = example["objects"]
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].squeeze(0)
    
    # Get ground truth
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
        "image": image
    }

def calculate_iou(pred_bbox, true_bbox):
    """Calculate IoU between predicted and true bounding boxes"""
    # Convert normalized coordinates to absolute
    pred_x1, pred_y1, pred_w, pred_h = pred_bbox
    true_x1, true_y1, true_w, true_h = true_bbox
    
    pred_x2 = pred_x1 + pred_w
    pred_y2 = pred_y1 + pred_h
    true_x2 = true_x1 + true_w
    true_y2 = true_y1 + true_h
    
    # Calculate intersection
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

def validate_dino_detector(model_path, test_dataset, processor):
    """Validate DINO detector model"""
    model = DinoDetector()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    correct_classifications = 0
    total_samples = 0
    iou_scores = []
    
    with torch.no_grad():
        for example in test_dataset:
            processed = preprocess_for_validation(example, processor)
            
            # Forward pass
            outputs = model(processed["pixel_values"].unsqueeze(0))
            
            # Get predictions
            pred_logits = outputs["logits"][0]
            pred_bbox = outputs["bbox"][0]
            
            # Classification accuracy
            pred_class = torch.argmax(pred_logits).item()
            true_class = processed["label"].item()
            
            if pred_class == true_class:
                correct_classifications += 1
            
            # IoU calculation
            pred_bbox_np = pred_bbox.cpu().numpy()
            true_bbox_np = processed["bbox"].cpu().numpy()
            iou = calculate_iou(pred_bbox_np, true_bbox_np)
            iou_scores.append(iou)
            
            total_samples += 1
    
    classification_acc = correct_classifications / total_samples
    mean_iou = np.mean(iou_scores)
    
    return {
        "classification_accuracy": classification_acc,
        "mean_iou": mean_iou,
        "total_samples": total_samples
    }

def validate_yolov8(test_dataset):
    """Validate using YOLOv8 (requires ultralytics)"""
    try:
        from ultralytics import YOLO
        
        # Load pretrained YOLOv8
        model = YOLO('yolov8n.pt')
        
        correct_classifications = 0
        total_samples = 0
        iou_scores = []
        
        for example in test_dataset:
            image = example["image"]
            annotations = example["objects"]
            
            if len(annotations["bbox"]) == 0:
                continue
                
            # Run inference
            results = model(image, verbose=False)
            
            if len(results[0].boxes) > 0:
                # Get first detection
                pred_box = results[0].boxes[0]
                pred_class = int(pred_box.cls.item())
                pred_bbox = pred_box.xywhn[0].cpu().numpy()  # normalized xywh
                
                # Ground truth
                true_class = annotations["category"][0]
                true_bbox = annotations["bbox"][0]
                w, h = image.size
                true_bbox_norm = [true_bbox[0]/w, true_bbox[1]/h, true_bbox[2]/w, true_bbox[3]/h]
                
                # Classification accuracy
                if pred_class == true_class:
                    correct_classifications += 1
                
                # IoU calculation
                iou = calculate_iou(pred_bbox, true_bbox_norm)
                iou_scores.append(iou)
            
            total_samples += 1
        
        classification_acc = correct_classifications / total_samples if total_samples > 0 else 0
        mean_iou = np.mean(iou_scores) if iou_scores else 0
        
        return {
            "classification_accuracy": classification_acc,
            "mean_iou": mean_iou,
            "total_samples": total_samples
        }
        
    except ImportError:
        print("YOLOv8 validation skipped - ultralytics not installed")
        return None

def validate_detectron2(test_dataset):
    """Validate using Detectron2"""
    try:
        import detectron2
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        
        # Setup Detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        
        correct_classifications = 0
        total_samples = 0
        iou_scores = []
        
        for example in test_dataset:
            image = example["image"]
            annotations = example["objects"]
            
            if len(annotations["bbox"]) == 0:
                continue
            
            # Convert PIL to numpy
            image_np = np.array(image)
            
            # Run inference
            outputs = predictor(image_np)
            
            if len(outputs["instances"]) > 0:
                # Get first detection
                pred_class = outputs["instances"].pred_classes[0].item()
                pred_bbox = outputs["instances"].pred_boxes[0].tensor[0].cpu().numpy()
                
                # Convert to normalized xywh
                h, w = image_np.shape[:2]
                pred_bbox_norm = [
                    pred_bbox[0]/w, pred_bbox[1]/h,
                    (pred_bbox[2]-pred_bbox[0])/w, (pred_bbox[3]-pred_bbox[1])/h
                ]
                
                # Ground truth
                true_class = annotations["category"][0]
                true_bbox = annotations["bbox"][0]
                true_bbox_norm = [true_bbox[0]/w, true_bbox[1]/h, true_bbox[2]/w, true_bbox[3]/h]
                
                # Classification accuracy
                if pred_class == true_class:
                    correct_classifications += 1
                
                # IoU calculation
                iou = calculate_iou(pred_bbox_norm, true_bbox_norm)
                iou_scores.append(iou)
            
            total_samples += 1
        
        classification_acc = correct_classifications / total_samples if total_samples > 0 else 0
        mean_iou = np.mean(iou_scores) if iou_scores else 0
        
        return {
            "classification_accuracy": classification_acc,
            "mean_iou": mean_iou,
            "total_samples": total_samples
        }
        
    except ImportError:
        print("Detectron2 validation skipped - detectron2 not installed")
        return None

def main():
    # Load test dataset
    test_dataset = load_dataset("detection-datasets/coco", split="val[:100]")
    
    # Initialize processor for DINO
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    print("Starting validation...")
    
    # Validate DINO detector
    if torch.cuda.is_available():
        print("Using GPU for validation")
    
    print("\n1. Validating DINO Detector...")
    dino_results = validate_dino_detector("dino2_detector.pth", test_dataset, processor)
    print(f"DINO Results:")
    print(f"  Classification Accuracy: {dino_results['classification_accuracy']:.4f}")
    print(f"  Mean IoU: {dino_results['mean_iou']:.4f}")
    print(f"  Total Samples: {dino_results['total_samples']}")
    
    # Validate YOLOv8
    print("\n2. Validating YOLOv8...")
    yolo_results = validate_yolov8(test_dataset)
    if yolo_results:
        print(f"YOLOv8 Results:")
        print(f"  Classification Accuracy: {yolo_results['classification_accuracy']:.4f}")
        print(f"  Mean IoU: {yolo_results['mean_iou']:.4f}")
        print(f"  Total Samples: {yolo_results['total_samples']}")
    
    # Validate Detectron2
    print("\n3. Validating Detectron2...")
    detectron_results = validate_detectron2(test_dataset)
    if detectron_results:
        print(f"Detectron2 Results:")
        print(f"  Classification Accuracy: {detectron_results['classification_accuracy']:.4f}")
        print(f"  Mean IoU: {detectron_results['mean_iou']:.4f}")
        print(f"  Total Samples: {detectron_results['total_samples']}")
    
    # Summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"DINO Detector    - Acc: {dino_results['classification_accuracy']:.4f}, IoU: {dino_results['mean_iou']:.4f}")
    if yolo_results:
        print(f"YOLOv8          - Acc: {yolo_results['classification_accuracy']:.4f}, IoU: {yolo_results['mean_iou']:.4f}")
    if detectron_results:
        print(f"Detectron2      - Acc: {detectron_results['classification_accuracy']:.4f}, IoU: {detectron_results['mean_iou']:.4f}")

if __name__ == "__main__":
    main()