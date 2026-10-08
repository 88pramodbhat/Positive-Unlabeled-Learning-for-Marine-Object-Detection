import os
try:
    import pytest
except ImportError:
    pytest = None

from src.config import Config

def test_default_config_loading():
    cfg = Config.load_yaml("configs/default_config.yaml")
    assert cfg.project_name == "positive-unlabeled-marine-detection"
    assert cfg.num_classes == 32
    assert len(cfg.class_names) == 32
    assert "pyrosome" in cfg.class_names
    assert "benthic worm" in cfg.class_names

def test_override_config_loading():
    cfg = Config.load_yaml("configs/default_config.yaml", "configs/stage2_pseudolabel.yaml")
    assert cfg.pseudo_label_conf_thresh == 0.70
    assert cfg.prior_pi == 0.35

if __name__ == "__main__":
    test_default_config_loading()
    test_override_config_loading()
    print("✓ test_config passed")
