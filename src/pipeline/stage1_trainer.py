import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..losses.focal_loss import FocalLoss

class Stage1Trainer:
    """
    Stage 1: Supervised Learning Trainer.
    Trains the detector on partially labeled underwater images.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.class_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.bbox_loss_fn = nn.SmoothL1Loss()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Stage 1 Train Epoch {epoch}")
        for images, targets in pbar:
            images = images.to(self.device)
            
            # Prepare batch targets
            batch_labels = []
            batch_bboxes = []
            for t in targets:
                if len(t["labels"]) > 0:
                    batch_labels.append(t["labels"][0].item())
                    batch_bboxes.append(t["boxes"][0])
                else:
                    batch_labels.append(0)
                    batch_bboxes.append(torch.zeros(4))

            labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
            bboxes_tensor = torch.stack(batch_bboxes).to(self.device)

            self.optimizer.zero_grad()
            logits, bbox_preds = self.model(images)

            cls_loss = self.class_loss_fn(logits, labels_tensor)
            box_loss = self.bbox_loss_fn(bbox_preds, bboxes_tensor)
            loss = cls_loss + 2.0 * box_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(self.train_loader))

    def run(self, epochs: int, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(epoch)
            print(f"[Stage 1] Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"[Stage 1] Checkpoint saved to {save_path}")
