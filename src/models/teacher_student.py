import copy
import torch
import torch.nn as nn
from typing import Dict, Any

class TeacherStudentWrapper(nn.Module):
    """
    Teacher-Student Semi-Supervised Model Wrapper.
    Teacher model creates reliable pseudo-labels on unlabeled data.
    Student model is trained on combined labeled and pseudo-labeled data.
    Teacher weights are updated smoothly using Exponential Moving Average (EMA).
    """
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        super().__init__()
        self.student = student_model
        self.teacher = copy.deepcopy(student_model)
        self.ema_decay = ema_decay
        
        # Freeze teacher gradients
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self):
        """
        Updates teacher model parameters via EMA:
        w_teacher = alpha * w_teacher + (1 - alpha) * w_student
        """
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

    def forward(self, x: torch.Tensor, is_teacher: bool = False):
        if is_teacher:
            with torch.no_grad():
                return self.teacher(x)
        return self.student(x)
