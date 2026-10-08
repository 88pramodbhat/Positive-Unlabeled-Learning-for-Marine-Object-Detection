import os
import tempfile
import torch
from torch.utils.data import DataLoader
try:
    import pytest
except ImportError:
    pytest = None

from src.models import MarineObjectDetector
from src.data import FathomNetDataset, get_underwater_transforms, collate_fn
from src.pipeline import PseudoLabeler
from scripts.prepare_fathomnet import create_synthetic_underwater_dataset

def test_pseudo_labeler_execution():
    with tempfile.TemporaryDirectory() as tmp_path:
        data_dir = os.path.join(tmp_path, "fathomnet")
        create_synthetic_underwater_dataset(data_dir)
        
        img_dir = os.path.join(data_dir, "train_unlabeled")
        transform = get_underwater_transforms(is_train=False)
        dataset = FathomNetDataset(img_dir, is_unlabeled=True, transform=transform)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        model = MarineObjectDetector(num_classes=32, backbone_type="resnet50", pretrained=False)
        output_json = os.path.join(tmp_path, "pseudo_out.json")
        
        labeler = PseudoLabeler(model, loader, conf_threshold=0.01, device="cpu")
        coco_data = labeler.generate(output_json)
        
        assert os.path.exists(output_json)
        assert "images" in coco_data
        assert "annotations" in coco_data

if __name__ == "__main__":
    test_pseudo_labeler_execution()
    print("✓ test_pseudo_labeler passed")
