import os
import torch
import gradio as gr
from PIL import Image
import numpy as np

from src.config import Config
from src.data import get_underwater_transforms
from src.models import MarineObjectDetector
from src.visualization import draw_bounding_boxes
from src.utils import load_checkpoint

# Load default configuration
cfg = Config.load_yaml("configs/default_config.yaml")
device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

# Load model
model = MarineObjectDetector(
    num_classes=cfg.num_classes,
    backbone_type=cfg.backbone,
    pretrained=False
)
weights_path = "checkpoints/stage2_best.pt"
if os.path.exists(weights_path):
    load_checkpoint(model, weights_path, device=device)

def run_marine_detection(input_image: Image.Image, conf_threshold: float):
    if input_image is None:
        return None, "Please upload an underwater image."

    transform = get_underwater_transforms(is_train=False)
    tensor_img = transform(input_image.convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        class_logits, bbox_preds = model(tensor_img)
        probs = torch.softmax(class_logits, dim=-1)
        scores, labels = torch.max(probs, dim=-1)

        score = scores[0].cpu().item()
        label = labels[0].cpu().item()
        box = bbox_preds[0].cpu().numpy() # [x1, y1, x2, y2] normalized

    if score >= conf_threshold:
        annotated_img = draw_bounding_boxes(
            image=input_image,
            boxes=[box],
            scores=[score],
            labels=[label],
            class_names=cfg.class_names
        )
        species = cfg.class_names[label] if label < len(cfg.class_names) else f"Species_{label}"
        x1, y1, x2, y2 = box
        info_str = f"✅ Detected Species: **{species}**\n- Confidence Score: **{score:.2%}**\n- Bounding Box (norm): `[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]`"
    else:
        annotated_img = input_image
        info_str = f"⚠️ No marine organisms detected above confidence threshold {conf_threshold:.2f}."

    return annotated_img, info_str

# Build Gradio Web Application interface
demo = gr.Interface(
    fn=run_marine_detection,
    inputs=[
        gr.Image(type="pil", label="Upload Underwater Marine Image"),
        gr.Slider(minimum=0.10, maximum=0.95, value=0.25, step=0.05, label="Detection Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Marine Object Detections"),
        gr.Markdown(label="Detection Results")
    ],
    title="🌊 Positive-Unlabeled Marine Object Detection",
    description=(
        "An AI-powered Positive-Unlabeled (PU) Semi-Supervised Learning detector for underwater marine species detection "
        "trained on the **FathomNet** dataset. Handles partial annotations, low-light blur, and class imbalance."
    ),
    examples=[],
    theme="glass",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
