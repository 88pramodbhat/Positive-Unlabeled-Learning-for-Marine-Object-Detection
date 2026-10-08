import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..models.teacher_student import TeacherStudentWrapper
from ..losses.pu_loss import NonNegativePULoss

class Stage2Trainer:
    """
    Stage 2: Teacher-Student Semi-Supervised PU Learning Trainer.
    Trains student model on combined labeled and pseudo-labeled data.
    Uses Non-Negative Positive-Unlabeled (nnPU) loss to handle missing annotations.
    """
    def __init__(
        self,
        teacher_student: TeacherStudentWrapper,
        combined_loader: DataLoader,
        lr: float = 0.0005,
        weight_decay: float = 0.0005,
        prior_pi: float = 0.35,
        device: str = "cuda"
    ):
        self.ts_model = teacher_student.to(device)
        self.combined_loader = combined_loader
        self.device = device
        
        self.optimizer = optim.AdamW(self.ts_model.student.parameters(), lr=lr, weight_decay=weight_decay)
        self.pu_loss_fn = NonNegativePULoss(prior_pi=prior_pi, nnpu_gamma=1.0)
        self.bbox_loss_fn = nn.SmoothL1Loss()

    def train_epoch(self, epoch: int) -> float:
        self.ts_model.student.train()
        total_loss = 0.0
        
        pbar = tqdm(self.combined_loader, desc=f"Stage 2 PU Epoch {epoch}")
        for images, targets in pbar:
            images = images.to(self.device)
            
            # Separate positive logits and unlabeled logits for PU Loss
            pos_mask = torch.tensor([len(t["labels"]) > 0 for t in targets], device=self.device)
            
            self.optimizer.zero_grad()
            logits, bbox_preds = self.ts_model.student(images)
            
            pos_logits = logits[pos_mask].view(-1) if pos_mask.sum() > 0 else torch.zeros(0, device=self.device)
            unlabeled_logits = logits[~pos_mask].view(-1) if (~pos_mask).sum() > 0 else torch.zeros(0, device=self.device)
            
            # Compute nnPU loss for classification
            pu_cls_loss = self.pu_loss_fn(pos_logits, unlabeled_logits)

            # Compute box loss for labeled targets
            batch_bboxes = []
            for t in targets:
                if len(t["boxes"]) > 0:
                    batch_bboxes.append(t["boxes"][0])
                else:
                    batch_bboxes.append(torch.zeros(4))
            bboxes_tensor = torch.stack(batch_bboxes).to(self.device)
            box_loss = self.bbox_loss_fn(bbox_preds, bboxes_tensor)

            loss = pu_cls_loss + 2.0 * box_loss

            loss.backward()
            self.optimizer.step()
            
            # Update Teacher model parameters via EMA
            self.ts_model.update_teacher()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(1, len(self.combined_loader))

    def run(self, epochs: int, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(epoch)
            print(f"[Stage 2 PU] Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.ts_model.student.state_dict(), save_path)
                print(f"[Stage 2 PU] Checkpoint saved to {save_path}")
