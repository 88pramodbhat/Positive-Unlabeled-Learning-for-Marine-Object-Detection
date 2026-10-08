import torch
try:
    import pytest
except ImportError:
    pytest = None

from src.losses.pu_loss import NonNegativePULoss

def test_pu_loss_computation():
    pu_loss_fn = NonNegativePULoss(prior_pi=0.35, nnpu_gamma=1.0)
    
    pos_logits = torch.tensor([2.5, 1.8, 3.1, 0.9])
    unlabeled_logits = torch.tensor([-1.2, 0.5, -0.8, 1.1, -2.0])
    
    loss = pu_loss_fn(pos_logits, unlabeled_logits)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0

def test_pu_loss_empty_inputs():
    pu_loss_fn = NonNegativePULoss(prior_pi=0.35)
    pos_logits = torch.zeros(0)
    unlabeled_logits = torch.tensor([1.0, -1.0])
    
    loss = pu_loss_fn(pos_logits, unlabeled_logits)
    assert not torch.isnan(loss)

if __name__ == "__main__":
    test_pu_loss_computation()
    test_pu_loss_empty_inputs()
    print("✓ test_pu_loss passed")
