import torch
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

from src.evaluation.metrics import calculate_map, compute_iou

def test_iou_calculation():
    box1 = np.array([0.0, 0.0, 10.0, 10.0])
    box2 = np.array([5.0, 0.0, 15.0, 10.0])
    
    iou = compute_iou(box1, box2)
    # Intersection = 5x10=50, Union = 100+100-50=150, IoU = 50/150 = 0.3333
    assert abs(iou - 0.3333) < 1e-3

def test_map_calculation():
    predictions = [
        {"boxes": torch.tensor([[0.0, 0.0, 0.5, 0.5]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([0])}
    ]
    targets = [
        {"boxes": torch.tensor([[0.0, 0.0, 0.5, 0.5]]), "labels": torch.tensor([0])}
    ]
    
    mean_ap, per_class_ap, (prec, rec, f1) = calculate_map(predictions, targets, num_classes=1, iou_thresh=0.5)
    assert mean_ap > 0.0
    assert prec == 1.0
    assert rec == 1.0

if __name__ == "__main__":
    test_iou_calculation()
    test_map_calculation()
    print("✓ test_metrics passed")
