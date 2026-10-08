import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import ResNet50Backbone, FPNBackbone
from typing import Dict, Any, Tuple, List

class MarineObjectDetector(nn.Module):
    """
    Positive-Unlabeled Marine Object Detection Neural Network.
    Integrates feature backbone (ResNet-50 / FPN), Region Proposal / Anchor heads,
    and output classification + bounding box regression branches.
    """
    def __init__(
        self,
        num_classes: int = 32,
        backbone_type: str = "resnet50",
        pretrained: bool = True,
        conf_thresh: float = 0.25
    ):
        super().__init__()
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh

        if backbone_type == "fpn":
            self.backbone = FPNBackbone(pretrained=pretrained, out_channels=256)
            in_channels = 256
        else:
            self.backbone = ResNet50Backbone(pretrained=pretrained)
            in_channels = self.backbone.out_channels

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head (32 marine species + 1 background/unknown)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Bounding box regression head [x1, y1, x2, y2]
        self.bbox_regressor = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Normalized coordinates [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        pooled = self.pool(features).squeeze(-1).squeeze(-1)
        
        class_logits = self.classifier(pooled)
        bbox_preds = self.bbox_regressor(pooled)
        
        return class_logits, bbox_preds

    def predict(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Runs inference on batch of images and returns filtered bounding boxes,
        confidence scores, and class predictions.
        """
        self.eval()
        with torch.no_grad():
            logits, bboxes = self.forward(x)
            scores, class_ids = torch.max(F.softmax(logits, dim=-1), dim=-1)
            
            results = []
            for b in range(x.size(0)):
                score = scores[b]
                cls_id = class_ids[b]
                bbox = bboxes[b]
                
                if score >= self.conf_thresh:
                    results.append({
                        "boxes": bbox.unsqueeze(0),
                        "scores": score.unsqueeze(0),
                        "labels": cls_id.unsqueeze(0)
                    })
                else:
                    results.append({
                        "boxes": torch.zeros((0, 4), device=x.device),
                        "scores": torch.zeros((0,), device=x.device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=x.device)
                    })
            return results
