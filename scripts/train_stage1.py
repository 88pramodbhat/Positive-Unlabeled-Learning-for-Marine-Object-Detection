import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data import FathomNetDataset, get_underwater_transforms, collate_fn
from src.models import MarineObjectDetector
from src.pipeline import Stage1Trainer
from src.utils import set_seed

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Supervised Learning for Marine Object Detection")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--override", type=str, default="configs/stage1_supervised.yaml", help="Override config")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    args = parser.parse_args()

    cfg = Config.load_yaml(args.config, args.override)
    if args.epochs:
        cfg.epochs = args.epochs

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    print(f"[Stage 1 CLI] Running on device: {device}")

    # Ensure dataset exists
    if not os.path.exists(cfg.labeled_anno):
        print(f"[Stage 1 CLI] Dataset annotations not found. Generating demo dataset...")
        from scripts.prepare_fathomnet import create_synthetic_underwater_dataset
        create_synthetic_underwater_dataset(cfg.data_dir)

    train_transform = get_underwater_transforms(is_train=True)
    val_transform = get_underwater_transforms(is_train=False)

    train_dataset = FathomNetDataset(cfg.labeled_images, cfg.labeled_anno, transform=train_transform)
    val_dataset = FathomNetDataset(cfg.test_images, cfg.test_anno, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MarineObjectDetector(
        num_classes=cfg.num_classes,
        backbone_type=cfg.backbone,
        pretrained=cfg.pretrained
    )

    trainer = Stage1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        device=device
    )

    save_checkpoint_path = os.path.join(cfg.checkpoint_dir, "stage1_best.pt")
    trainer.run(epochs=cfg.epochs, save_path=save_checkpoint_path)

if __name__ == "__main__":
    main()
