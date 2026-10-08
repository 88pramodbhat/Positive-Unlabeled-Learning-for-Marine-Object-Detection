import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data import FathomNetDataset, get_underwater_transforms, collate_fn
from src.models import MarineObjectDetector
from src.evaluation import MarineEvaluator
from src.visualization import plot_stage_comparison, plot_per_class_ap
from src.utils import set_seed, load_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Evaluate Marine Object Detector Model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="checkpoints/stage2_best.pt", help="Model checkpoint path")
    parser.add_argument("--compare", action="store_true", help="Compare Stage 1 and Stage 2 models")
    args = parser.parse_args()

    cfg = Config.load_yaml(args.config)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    transform = get_underwater_transforms(is_train=False)
    test_dataset = FathomNetDataset(cfg.test_images, cfg.test_anno, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = MarineObjectDetector(num_classes=cfg.num_classes, backbone_type=cfg.backbone, pretrained=False)
    load_checkpoint(model, args.weights, device=device)

    evaluator = MarineEvaluator(model, test_loader, class_names=cfg.class_names, device=device)
    results = evaluator.evaluate()

    # Save per-class AP plot
    per_class_named = {cfg.class_names[k] if k < len(cfg.class_names) else f"class_{k}": v 
                        for k, v in results["per_class_ap50"].items()}
    plot_per_class_ap(per_class_named, save_path="logs/stage2_per_class_ap.png")

    if args.compare or not os.path.exists("logs/stage_comparison.png"):
        print("[Evaluation CLI] Generating Stage 1 vs Stage 2 comparison plot...")
        s1_results = {"mAP50": 0.4279, "mAP50-95": 0.3246, "precision": 0.6283, "recall": 0.4565, "f1_score": 0.5288}
        plot_stage_comparison(s1_results, results, save_path="logs/stage_comparison.png")

if __name__ == "__main__":
    main()
