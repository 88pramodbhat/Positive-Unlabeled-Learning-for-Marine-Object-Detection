import os
import torch
try:
    import pytest
except ImportError:
    pytest = None

from src.data import FathomNetDataset, get_underwater_transforms
from scripts.prepare_fathomnet import create_synthetic_underwater_dataset

def test_dataset_loading():
    data_dir = "data/test_fathomnet"
    create_synthetic_underwater_dataset(data_dir)
    anno_file = os.path.join(data_dir, "annotations/train_labeled.json")
    img_dir = os.path.join(data_dir, "train_labeled")
    
    transform = get_underwater_transforms(is_train=False)
    dataset = FathomNetDataset(img_dir, anno_file, transform=transform)
    
    assert len(dataset) > 0
    img, target = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 640, 640)
    assert "boxes" in target
    assert "labels" in target

if __name__ == "__main__":
    test_dataset_loading()
    print("✓ test_dataset passed")
