import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from PIL import Image

from src.config import Config
from src.data import get_underwater_transforms
from src.models import MarineObjectDetector
from src.visualization import draw_bounding_boxes
from src.utils import load_checkpoint

def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    output_path: str,
    class_names: list,
    conf_thresh: float = 0.25,
    device: str = "cuda"
):
    if not os.path.exists(image_path):
        print(f"[Predict] Error: Image file '{image_path}' not found.")
        return

    raw_image = Image.open(image_path).convert("RGB")
    transform = get_underwater_transforms(is_train=False)
    tensor_img = transform(raw_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        class_logits, bbox_preds = model(tensor_img)
        probs = torch.softmax(class_logits, dim=-1)
        scores, labels = torch.max(probs, dim=-1)

        score = scores[0].cpu().item()
        label = labels[0].cpu().item()
        box = bbox_preds[0].cpu().numpy() # normalized [x1, y1, x2, y2]

    if score >= conf_thresh:
        annotated_img = draw_bounding_boxes(
            image=raw_image,
            boxes=[box],
            scores=[score],
            labels=[label],
            class_names=class_names
        )
        species = class_names[label] if label < len(class_names) else f"Species_{label}"
        print(f"[Predict] Detected: {species} (Confidence: {score:.2f}) in '{image_path}'")
    else:
        annotated_img = raw_image
        print(f"[Predict] No marine objects detected above confidence threshold {conf_thresh}.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    annotated_img.save(output_path)
    print(f"[Predict] Result saved to '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Marine Object Detection Image Inference")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="checkpoints/stage2_best.pt", help="Model weights path")
    parser.add_argument("--image", type=str, required=True, help="Input image path or directory")
    parser.add_argument("--output", type=str, default="logs/predictions/output.jpg", help="Output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    cfg = Config.load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    model = MarineObjectDetector(num_classes=cfg.num_classes, backbone_type=cfg.backbone, pretrained=False)
    load_checkpoint(model, args.weights, device=device)

    if os.path.isfile(args.image):
        predict_single_image(model, args.image, args.output, cfg.class_names, args.conf, device)
    elif os.path.isdir(args.image):
        files = [f for f in os.listdir(args.image) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            in_p = os.path.join(args.image, f)
            out_p = os.path.join(os.path.dirname(args.output), f"pred_{f}")
            predict_single_image(model, in_p, out_p, cfg.class_names, args.conf, device)

if __name__ == "__main__":
    main()
