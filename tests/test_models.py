import torch
try:
    import pytest
except ImportError:
    pytest = None

from src.models import MarineObjectDetector, TeacherStudentWrapper

def test_marine_detector_forward():
    model = MarineObjectDetector(num_classes=32, backbone_type="resnet50", pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    
    class_logits, bbox_preds = model(x)
    assert class_logits.shape == (2, 32)
    assert bbox_preds.shape == (2, 4)
    assert (bbox_preds >= 0.0).all() and (bbox_preds <= 1.0).all()

def test_teacher_student_ema_update():
    student = MarineObjectDetector(num_classes=32, backbone_type="resnet50", pretrained=False)
    ts_wrapper = TeacherStudentWrapper(student, ema_decay=0.9)
    
    # Mutate student parameter
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
            
    ts_wrapper.update_teacher()
    # Teacher params should update toward student params
    for s_p, t_p in zip(student.parameters(), ts_wrapper.teacher.parameters()):
        assert not torch.allclose(s_p, t_p)

if __name__ == "__main__":
    test_marine_detector_forward()
    test_teacher_student_ema_update()
    print("✓ test_models passed")
