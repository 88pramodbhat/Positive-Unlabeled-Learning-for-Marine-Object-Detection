import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw

def create_synthetic_underwater_dataset(data_dir: str = "data/fathomnet"):
    """
    Generates a synthetic demo dataset mirroring FathomNet underwater dataset layout
    with 32 marine species classes for out-of-the-box execution and testing.
    """
    print(f"[Dataset Preparation] Setting up dataset structure in '{data_dir}'...")
    
    train_labeled_dir = os.path.join(data_dir, "train_labeled")
    train_unlabeled_dir = os.path.join(data_dir, "train_unlabeled")
    test_dir = os.path.join(data_dir, "test")
    anno_dir = os.path.join(data_dir, "annotations")
    
    for d in [train_labeled_dir, train_unlabeled_dir, test_dir, anno_dir]:
        os.makedirs(d, exist_ok=True)

    species_list = [
        "pyrosome", "larvacean", "urchin", "crab", "soft coral", "barnacle", "sea pen",
        "sea slug", "octopus", "jellyfish", "sea star", "sea cucumber", "brittle star",
        "sponge", "anemone", "bony fish", "shark", "ray", "shrimp", "amphipod",
        "isopod", "bivalve", "gastropod", "hydroid", "coral", "tunicates", "bryozoan",
        "ctenophore", "polychaete", "benthic worm", "sea snail", "sea squirt"
    ]
    categories = [{"id": i, "name": name} for i, name in enumerate(species_list)]

    def generate_images(img_dir, count, is_labeled=True):
        images_info = []
        annotations_info = []
        anno_id = 1

        for i in range(1, count + 1):
            filename = f"marine_{i:04d}.jpg"
            img_path = os.path.join(img_dir, filename)
            
            # Create synthetic underwater image with deep blue/green background
            bg_color = (np.random.randint(5, 30), np.random.randint(40, 80), np.random.randint(60, 120))
            img = Image.new("RGB", (640, 640), bg_color)
            draw = ImageDraw.Draw(img)

            # Add random ocean texture noise
            noise = np.random.randint(0, 30, (640, 640, 3), dtype=np.uint8)
            img_np = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img)

            images_info.append({"id": i, "file_name": filename, "width": 640, "height": 640})

            if is_labeled:
                num_objects = np.random.randint(1, 4)
                for _ in range(num_objects):
                    cls_id = np.random.randint(0, len(species_list))
                    x1 = np.random.randint(50, 450)
                    y1 = np.random.randint(50, 450)
                    w = np.random.randint(50, 150)
                    h = np.random.randint(50, 150)

                    # Draw colorful synthetic marine object shape
                    obj_color = (np.random.randint(150, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                    draw.ellipse([x1, y1, x1 + w, y1 + h], fill=obj_color, outline=(255, 255, 255))

                    annotations_info.append({
                        "id": anno_id,
                        "image_id": i,
                        "category_id": cls_id,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    anno_id += 1

            img.save(img_path, quality=90)

        return {"images": images_info, "annotations": annotations_info, "categories": categories}

    print("Generating synthetic labeled training images...")
    labeled_data = generate_images(train_labeled_dir, count=20, is_labeled=True)
    with open(os.path.join(anno_dir, "train_labeled.json"), "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=2)

    print("Generating synthetic unlabeled training images...")
    _ = generate_images(train_unlabeled_dir, count=20, is_labeled=False)

    print("Generating synthetic test images...")
    test_data = generate_images(test_dir, count=10, is_labeled=True)
    with open(os.path.join(anno_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"[Dataset Preparation] Done! Created dataset in '{data_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FathomNet Marine Dataset")
    parser.add_argument("--data_dir", type=str, default="data/fathomnet", help="Dataset directory")
    args = parser.parse_args()

    create_synthetic_underwater_dataset(args.data_dir)
