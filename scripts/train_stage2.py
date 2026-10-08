import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.config import Config
from src.data import FathomNetDataset, get_underwater_transforms, collate_fn
from src.models import MarineObjectDetector, TeacherStudentWrapper
from src.pipeline import Stage2Trainer
from src.utils import set_seed, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Teacher-Student PU Semi-Supervised Learning")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--override", type=str, default="configs/stage2_pseudolabel.yaml", help="Override config")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    args = parser.parse_args()

    cfg = Config.load_yaml(args.config, args.override)
    if args.epochs:
        cfg.epochs = args.epochs

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    print(f"[Stage 2 PU CLI] Running on device: {device}")

    # Ensure pseudo-labels exist or generate them
    if not os.path.exists(cfg.pseudo_anno):
        print("[Stage 2 PU CLI] Pseudo-labels not found. Generating pseudo-labels...")
        from scripts.generate_pseudo_labels import main as gen_pseudo_main
        gen_pseudo_main()

    transform = get_underwater_transforms(is_train=True)
    
    labeled_dataset = FathomNetDataset(cfg.labeled_images, cfg.labeled_anno, transform=transform)
    pseudo_dataset = FathomNetDataset(cfg.unlabeled_images, cfg.pseudo_anno, transform=transform)
    
    combined_dataset = ConcatDataset([labeled_dataset, pseudo_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    student_base = MarineObjectDetector(
        num_classes=cfg.num_classes,
        backbone_type=cfg.backbone,
        pretrained=cfg.pretrained
    )
    load_checkpoint(student_base, "checkpoints/stage1_best.pt", device=device)

    ts_model = TeacherStudentWrapper(student_base, ema_decay=cfg.ema_decay)

    trainer = Stage2Trainer(
        teacher_student=ts_model,
        combined_loader=combined_loader,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        prior_pi=cfg.prior_pi,
        device=device
    )

    save_path = os.path.join(cfg.checkpoint_dir, "stage2_best.pt")
    trainer.run(epochs=cfg.epochs, save_path=save_path)

if __name__ == "__main__":
    main()
