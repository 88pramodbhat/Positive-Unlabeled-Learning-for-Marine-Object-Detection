import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data import FathomNetDataset, get_underwater_transforms, collate_fn
from src.models import MarineObjectDetector
from src.pipeline import PseudoLabeler
from src.utils import set_seed, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Generate Pseudo-Labels for Unlabeled Marine Images")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--override", type=str, default="configs/stage2_pseudolabel.yaml", help="Override config")
    parser.add_argument("--teacher_weights", type=str, default="checkpoints/stage1_best.pt", help="Teacher model weights")
    args = parser.parse_args()

    cfg = Config.load_yaml(args.config, args.override)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    print(f"[Pseudo-Label Generator] Using Teacher model weights from {args.teacher_weights}")

    transform = get_underwater_transforms(is_train=False)
    unlabeled_dataset = FathomNetDataset(cfg.unlabeled_images, transform=transform, is_unlabeled=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    teacher_model = MarineObjectDetector(
        num_classes=cfg.num_classes,
        backbone_type=cfg.backbone,
        pretrained=False
    )

    if not load_checkpoint(teacher_model, args.teacher_weights, device=device):
        print(f"[Pseudo-Label Generator] Note: Initializing randomly since checkpoint was not found.")

    labeler = PseudoLabeler(
        model=teacher_model,
        dataloader=unlabeled_loader,
        conf_threshold=cfg.pseudo_label_conf_thresh,
        device=device
    )

    labeler.generate(cfg.pseudo_anno)

if __name__ == "__main__":
    main()
