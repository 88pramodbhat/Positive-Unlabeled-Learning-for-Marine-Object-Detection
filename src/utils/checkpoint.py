import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

def save_checkpoint(
    state: Dict[str, Any],
    filepath: str = "checkpoints/model_checkpoint.pt"
):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"[Checkpoint] Saved checkpoint state to {filepath}")

def load_checkpoint(
    model: nn.Module,
    filepath: str,
    device: str = "cuda"
) -> bool:
    if not os.path.exists(filepath):
        print(f"[Checkpoint] Warning: File {filepath} not found.")
        return False
        
    state_dict = torch.load(filepath, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
        
    print(f"[Checkpoint] Successfully loaded weights from {filepath}")
    return True
