import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List

class PseudoLabeler:
    """
    Generates high-confidence pseudo-labels on unlabeled underwater marine images
    using the trained Teacher model.
    Low-confidence detections are filtered out to suppress label noise.
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        conf_threshold: float = 0.70,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.conf_threshold = conf_threshold
        self.device = device

    def generate(self, output_coco_file: str) -> Dict[str, Any]:
        self.model.eval()
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        anno_id = 1
        print(f"[PseudoLabeler] Generating pseudo-labels with conf threshold = {self.conf_threshold}...")
        
        with torch.no_grad():
            for idx, (images, targets) in enumerate(tqdm(self.dataloader, desc="Pseudo-Labeling")):
                images = images.to(self.device)
                
                # Predict class logits and bounding boxes
                class_logits, bbox_preds = self.model(images)
                probs = torch.softmax(class_logits, dim=-1)
                max_probs, class_ids = torch.max(probs, dim=-1)
                
                for b in range(images.size(0)):
                    target = targets[b]
                    file_name = target["file_name"]
                    img_id = target["image_id"].item()
                    
                    coco_data["images"].append({
                        "id": img_id,
                        "file_name": file_name,
                        "width": 640,
                        "height": 640
                    })
                    
                    conf = max_probs[b].item()
                    cls_id = class_ids[b].item()
                    bbox = bbox_preds[b].cpu().numpy() # [x1, y1, x2, y2] normalized
                    
                    # Store as pseudo-label if confidence exceeds threshold
                    if conf >= self.conf_threshold:
                        x1, y1, x2, y2 = bbox
                        w = max(0.01, x2 - x1) * 640
                        h = max(0.01, y2 - y1) * 640
                        x1_px = x1 * 640
                        y1_px = y1 * 640
                        
                        coco_data["annotations"].append({
                            "id": anno_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [round(float(x1_px), 2), round(float(y1_px), 2), round(float(w), 2), round(float(h), 2)],
                            "area": round(float(w * h), 2),
                            "score": round(float(conf), 4),
                            "iscrowd": 0,
                            "is_pseudo": True
                        })
                        anno_id += 1

        # Save generated pseudo-labels to JSON
        os.makedirs(os.path.dirname(output_coco_file), exist_ok=True)
        with open(output_coco_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        print(f"[PseudoLabeler] Successfully generated {len(coco_data['annotations'])} pseudo-labels saved to {output_coco_file}")
        return coco_data
