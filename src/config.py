import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Config:
    project_name: str = "positive-unlabeled-marine-detection"
    seed: int = 42
    device: str = "cuda"
    
    # Dataset config
    data_dir: str = "data/fathomnet"
    labeled_images: str = "data/fathomnet/train_labeled"
    unlabeled_images: str = "data/fathomnet/train_unlabeled"
    test_images: str = "data/fathomnet/test"
    labeled_anno: str = "data/fathomnet/annotations/train_labeled.json"
    test_anno: str = "data/fathomnet/annotations/test.json"
    pseudo_anno: str = "data/fathomnet/annotations/pseudo_labels.json"
    img_size: List[int] = field(default_factory=lambda: [640, 640])
    num_classes: int = 32
    class_names: List[str] = field(default_factory=list)
    
    # Model config
    backbone: str = "resnet50"
    pretrained: bool = True
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Training config
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    epochs: int = 25
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Teacher-Student & PU Learning
    prior_pi: float = 0.35
    nnpu_gamma: float = 1.0
    pseudo_label_conf_thresh: float = 0.70
    ema_decay: float = 0.999
    
    @classmethod
    def load_yaml(cls, yaml_path: str, override_path: Optional[str] = None) -> "Config":
        config = cls()
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                config._update_from_dict(data)

        if override_path and os.path.exists(override_path):
            with open(override_path, "r", encoding="utf-8") as f:
                override_data = yaml.safe_load(f) or {}
                config._update_from_dict(override_data)
                
        # Load class names if class_config specified
        class_cfg_path = getattr(config, "class_config", "configs/fathomnet_32classes.yaml")
        if os.path.exists(class_cfg_path):
            with open(class_cfg_path, "r", encoding="utf-8") as f:
                class_data = yaml.safe_load(f) or {}
                if "class_names" in class_data:
                    config.class_names = class_data["class_names"]
                    config.num_classes = len(config.class_names)
                    
        return config

    def _update_from_dict(self, data: Dict[str, Any]):
        for k, v in data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if hasattr(self, sub_k):
                        setattr(self, sub_k, sub_v)
            else:
                if hasattr(self, k):
                    setattr(self, k, v)
