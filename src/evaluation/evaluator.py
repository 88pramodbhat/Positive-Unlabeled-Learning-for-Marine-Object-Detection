import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List
from .metrics import calculate_map

class MarineEvaluator:
    """
    Evaluator for Marine Object Detection models.
    Computes global performance metrics (mAP50, mAP50-95, Precision, Recall, F1)
    and class breakdown (Top-5 and Bottom-5 marine species).
    """
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        class_names: List[str],
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device

    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        print("[MarineEvaluator] Evaluating model performance on test set...")
        with torch.no_grad():
            for images, targets in tqdm(self.dataloader, desc="Evaluation"):
                images = images.to(self.device)
                
                # Model inference
                class_logits, bbox_preds = self.model(images)
                probs = torch.softmax(class_logits, dim=-1)
                scores, labels = torch.max(probs, dim=-1)

                for b in range(images.size(0)):
                    all_predictions.append({
                        "boxes": bbox_preds[b].cpu(),
                        "scores": scores[b].cpu(),
                        "labels": labels[b].cpu()
                    })
                    all_targets.append({
                        "boxes": targets[b]["boxes"].cpu(),
                        "labels": targets[b]["labels"].cpu()
                    })

        # Calculate mAP at IoU 0.50
        num_classes = len(self.class_names)
        map50, per_class_ap50, (prec, rec, f1) = calculate_map(
            all_predictions, all_targets, num_classes=num_classes, iou_thresh=0.50
        )
        
        # Calculate mAP at IoU 0.75 for mAP50-95 estimation
        map75, _, _ = calculate_map(
            all_predictions, all_targets, num_classes=num_classes, iou_thresh=0.75
        )
        map50_95 = round((map50 + map75) / 2.0, 4)

        # Sort classes by AP50
        sorted_ap = sorted(
            [(self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}", ap) 
             for cls_id, ap in per_class_ap50.items()],
            key=lambda x: x[1],
            reverse=True
        )

        top_5 = sorted_ap[:5]
        bottom_5 = sorted_ap[-5:]

        results = {
            "mAP50": map50,
            "mAP50-95": map50_95,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "per_class_ap50": per_class_ap50,
            "top_5_classes": top_5,
            "bottom_5_classes": bottom_5
        }

        self.print_summary(results)
        return results

    def print_summary(self, results: Dict[str, Any]):
        print("\n" + "="*50)
        print("         MARINE OBJECT DETECTION EVALUATION SUMMARY")
        print("="*50)
        print(f"  mAP50     : {results['mAP50']:.4f}")
        print(f"  mAP50-95  : {results['mAP50-95']:.4f}")
        print(f"  Precision : {results['precision']:.4f}")
        print(f"  Recall    : {results['recall']:.4f}")
        print(f"  F1-Score  : {results['f1_score']:.4f}")
        print("-" * 50)
        print(" Top-5 Species by AP50:")
        for species, score in results['top_5_classes']:
            print(f"   - {species:<20}: {score:.4f}")
        print("-" * 50)
        print(" Bottom-5 Species by AP50:")
        for species, score in results['bottom_5_classes']:
            print(f"   - {species:<20}: {score:.4f}")
        print("="*50 + "\n")
