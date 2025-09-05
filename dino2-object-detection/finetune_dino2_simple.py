import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset
from PIL import Image
import numpy as np

class DinoDetector(nn.Module):
    def __init__(self, num_classes=91):  # COCO has 91 classes
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Linear(768, num_classes)
        self.bbox_head = nn.Linear(768, 4)
        
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # CLS token
        
        logits = self.classifier(cls_token)
        bbox = self.bbox_head(cls_token)
        
        return {"logits": logits, "bbox": bbox}

def preprocess_coco(example, processor):
    image = example["image"]
    annotations = example["objects"]
    

    
    # Process image
    try:
        inputs = processor(images=image, return_tensors="pt")

        pixel_values = inputs["pixel_values"].squeeze(0)
    except Exception as e:

        pixel_values = torch.zeros(3, 224, 224)
    
    # Get first bounding box and label for simplicity
    if isinstance(annotations, dict) and "bbox" in annotations and len(annotations["bbox"]) > 0:
        bbox = annotations["bbox"][0]  # [x, y, width, height]
        label = annotations["category"][0]
        
        # Normalize bbox to [0, 1]
        w, h = image.size
        bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]

    else:
        bbox_norm = [0, 0, 0, 0]
        label = 0

    
    result = {
        "pixel_values": pixel_values,
        "bbox": bbox_norm,  # Keep as list to avoid 0-dim tensor issues
        "label": int(label)  # Keep as int to avoid 0-dim tensor issues
    }

    print(f"Preprocessing result: pixel_values type={type(pixel_values)}, is_tensor={torch.is_tensor(pixel_values)}")
    return result

def collate_fn(batch):
    pixel_values = []
    bboxes = []
    labels = []
    
    for i, item in enumerate(batch):
        pv = item["pixel_values"]

        
        print(f"Collate item {i}: type={type(pv)}, len={len(pv) if isinstance(pv, list) else 'N/A'}")
        
        # Extract tensor from list
        if isinstance(pv, list) and len(pv) > 0:
            print(f"  First element: type={type(pv[0])}, is_tensor={torch.is_tensor(pv[0])}")
            pixel_values.append(pv[0])
        else:
            pixel_values.append(pv)
        bbox = item["bbox"]
        label = item["label"]
        
        # Convert to tensors if needed
        if isinstance(bbox, list):
            bboxes.append(torch.tensor(bbox, dtype=torch.float32))
        else:
            bboxes.append(bbox)
            
        if isinstance(label, (int, float)):
            labels.append(torch.tensor(label, dtype=torch.long))
        else:
            labels.append(label)

    
    # Final check - convert any remaining lists to tensors
    final_pixel_values = []
    tensor_count = 0
    list_tensor_count = 0
    dummy_count = 0
    
    for pv in pixel_values:
        if torch.is_tensor(pv):
            final_pixel_values.append(pv)
            tensor_count += 1
        elif isinstance(pv, list) and len(pv) > 0 and torch.is_tensor(pv[0]):
            final_pixel_values.append(pv[0])
            list_tensor_count += 1
        else:
            final_pixel_values.append(torch.zeros(3, 224, 224))
            dummy_count += 1
    
    print(f"Batch stats: {tensor_count} direct tensors, {list_tensor_count} from lists, {dummy_count} dummies")
    
    return {
        "pixel_values": torch.stack(final_pixel_values),
        "bboxes": torch.stack(bboxes),
        "labels": torch.stack(labels)
    }

def compute_loss(outputs, targets):
    logits = outputs["logits"]
    bbox_pred = outputs["bbox"]
    
    labels = targets["labels"]
    bbox_true = targets["bboxes"]
    
    print(f"Loss computation - logits shape: {logits.shape}, labels shape: {labels.shape}")
    print(f"bbox_pred shape: {bbox_pred.shape}, bbox_true shape: {bbox_true.shape}")
    
    # Classification loss
    cls_loss = nn.CrossEntropyLoss()(logits, labels)
    
    # Bbox regression loss (L1)
    bbox_loss = nn.L1Loss()(bbox_pred, bbox_true)
    
    return cls_loss + bbox_loss

def main():
    # Load COCO dataset (small subset)
    dataset = load_dataset("detection-datasets/coco", split="train[:1000]")
    
    # Initialize processor and model
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = DinoDetector()
    
    # Create a simple list-based dataset
    processed_data = []
    for i in range(len(dataset)):
        example = dataset[i]
        processed_example = preprocess_coco(example, processor)
        processed_data.append(processed_example)
    
    # Convert to simple dataset
    from torch.utils.data import Dataset as TorchDataset
    
    class SimpleDataset(TorchDataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    processed_dataset = SimpleDataset(processed_data)
    
    # Create dataloader
    dataloader = DataLoader(
        processed_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Training loop
    for epoch in range(5):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(batch["pixel_values"])
            loss = compute_loss(outputs, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} completed. Average loss: {total_loss/len(dataloader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "dino2_detector.pth")
    print("Model saved as dino2_detector.pth")

if __name__ == "__main__":
    main()