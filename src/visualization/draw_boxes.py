import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple

# Vibrant color palette for marine species visualization
COLOR_PALETTE = [
    (0, 255, 127), (255, 69, 0), (30, 144, 255), (255, 215, 0),
    (238, 130, 238), (0, 255, 255), (255, 105, 180), (124, 252, 0),
    (255, 165, 0), (138, 43, 226), (0, 191, 255), (50, 205, 50)
]

def draw_bounding_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None
) -> Image.Image:
    """
    Draws bounding boxes and marine species labels on an underwater image using PIL.
    """
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    img_w, img_h = img_copy.size

    for i, box in enumerate(boxes):
        if len(box) < 4:
            continue

        x1, y1, x2, y2 = box
        # Scale if normalized
        if x2 <= 1.0 and y2 <= 1.0:
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cls_id = labels[i] if labels is not None and i < len(labels) else 0
        score = scores[i] if scores is not None and i < len(scores) else 1.0
        
        box_color = color if color else COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Label text
        species_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"Species_{cls_id}"
        label_text = f"{species_name}: {score:.2f}" if scores is not None else species_name

        # Draw text background rectangle
        draw.rectangle([x1, max(0, y1 - 18), x1 + len(label_text) * 8 + 6, max(18, y1)], fill=box_color)
        draw.text((x1 + 3, max(0, y1 - 16)), label_text, fill=(0, 0, 0))

    return img_copy
