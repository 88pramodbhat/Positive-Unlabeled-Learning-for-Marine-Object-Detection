import numpy as np
import torch
from typing import Dict, List, Tuple, Any

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def calculate_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)

def calculate_map(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    num_classes: int = 32,
    iou_thresh: float = 0.50
) -> Tuple[float, Dict[int, float], Tuple[float, float, float]]:
    """
    Computes Mean Average Precision (mAP) at IoU threshold and per-class AP scores.
    """
    per_class_ap = {}
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for cls_id in range(num_classes):
        # Extract predictions for current class
        cls_preds = []
        for img_idx, pred in enumerate(predictions):
            labels = pred["labels"].cpu().numpy() if isinstance(pred["labels"], torch.Tensor) else np.array(pred["labels"])
            scores = pred["scores"].cpu().numpy() if isinstance(pred["scores"], torch.Tensor) else np.array(pred["scores"])
            boxes = pred["boxes"].cpu().numpy() if isinstance(pred["boxes"], torch.Tensor) else np.array(pred["boxes"])
            
            for l, s, b in zip(labels, scores, boxes):
                if l == cls_id:
                    cls_preds.append((img_idx, s, b))
                    
        # Extract targets for current class
        cls_gt = {}
        gt_count = 0
        for img_idx, tgt in enumerate(targets):
            labels = tgt["labels"].cpu().numpy() if isinstance(tgt["labels"], torch.Tensor) else np.array(tgt["labels"])
            boxes = tgt["boxes"].cpu().numpy() if isinstance(tgt["boxes"], torch.Tensor) else np.array(tgt["boxes"])
            
            cls_gt[img_idx] = []
            for l, b in zip(labels, boxes):
                if l == cls_id:
                    cls_gt[img_idx].append({"box": b, "matched": False})
                    gt_count += 1

        if len(cls_preds) == 0:
            per_class_ap[cls_id] = 0.0
            total_fn += gt_count
            continue
            
        # Sort predictions by descending confidence score
        cls_preds.sort(key=lambda x: x[1], reverse=True)
        
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        
        for i, (img_idx, score, pred_box) in enumerate(cls_preds):
            gts = cls_gt.get(img_idx, [])
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gts):
                iou = compute_iou(pred_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= iou_thresh and best_gt_idx >= 0 and not gts[best_gt_idx]["matched"]:
                tp[i] = 1.0
                gts[best_gt_idx]["matched"] = True
            else:
                fp[i] = 1.0

        tp_sum = np.sum(tp)
        fp_sum = np.sum(fp)
        fn_sum = gt_count - tp_sum
        
        total_tp += int(tp_sum)
        total_fp += int(fp_sum)
        total_fn += int(fn_sum)

        # Compute precision-recall curve & area under curve (AP)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (gt_count + 1e-6)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
        
        # P-R Curve area calculation with standard boundary padding
        r_padded = np.concatenate(([0.0], recalls, [1.0]))
        p_padded = np.concatenate(([1.0], precisions, [0.0]))
        # Make precision monotonically decreasing
        for p_i in range(len(p_padded) - 2, -1, -1):
            p_padded[p_i] = max(p_padded[p_i], p_padded[p_i + 1])
            
        ap = np.trapz(p_padded, r_padded) if len(r_padded) > 0 else 0.0
        per_class_ap[cls_id] = round(float(ap), 4)

    mean_ap = round(float(np.mean(list(per_class_ap.values()))), 4)
    prec, rec, f1 = calculate_precision_recall_f1(total_tp, total_fp, total_fn)
    
    return mean_ap, per_class_ap, (prec, rec, f1)
