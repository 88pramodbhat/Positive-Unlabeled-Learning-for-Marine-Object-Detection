import torch
import torch.nn as nn
import torch.nn.functional as F

class NonNegativePULoss(nn.Module):
    """
    Non-Negative Positive-Unlabeled (nnPU) Risk Estimator Loss.
    Solves the problem where unlabeled marine organisms are treated as background noise.
    
    Formula:
        R_pu = pi * R_p^+ + max(0, R_u^- - pi * R_p^-)
        
    Args:
        prior_pi (float): Class prior probability for positive marine species (default 0.35).
        nnpu_gamma (float): Penalty coefficient when negative risk bound is violated.
        loss_func (str): Underlying binary loss function ('sigmoid' or 'bce').
    """
    def __init__(self, prior_pi: float = 0.35, nnpu_gamma: float = 1.0, loss_func: str = "sigmoid"):
        super().__init__()
        self.pi = prior_pi
        self.gamma = nnpu_gamma
        self.loss_func = loss_func

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_func == "sigmoid":
            return torch.sigmoid(-logits * target)
        else:
            probs = torch.sigmoid(logits)
            return F.binary_cross_entropy(probs, target, reduction="none")

    def forward(
        self,
        pos_logits: torch.Tensor,
        unlabeled_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_logits: Logits from labeled positive marine objects.
            unlabeled_logits: Logits from unlabeled image regions.
        Returns:
            Computed nnPU scalar loss tensor.
        """
        if pos_logits.numel() == 0:
            pos_loss_p = torch.tensor(0.0, device=unlabeled_logits.device)
            pos_loss_n = torch.tensor(0.0, device=unlabeled_logits.device)
        else:
            pos_loss_p = self._loss(pos_logits, torch.ones_like(pos_logits)).mean()
            pos_loss_n = self._loss(pos_logits, -torch.ones_like(pos_logits)).mean()

        if unlabeled_logits.numel() == 0:
            unlabeled_loss_n = torch.tensor(0.0, device=pos_logits.device)
        else:
            unlabeled_loss_n = self._loss(unlabeled_logits, -torch.ones_like(unlabeled_logits)).mean()

        # Non-negative PU Risk Estimator Calculation
        negative_risk = unlabeled_loss_n - self.pi * pos_loss_n
        
        if negative_risk >= 0:
            pu_risk = self.pi * pos_loss_p + negative_risk
        else:
            # Apply non-negative constraint penalty
            pu_risk = self.pi * pos_loss_p - self.gamma * negative_risk

        return pu_risk
