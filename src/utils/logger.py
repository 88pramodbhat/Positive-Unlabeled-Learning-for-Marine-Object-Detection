import os
import logging
from typing import Dict, Any

def setup_logger(name: str = "PUMarineDetection", log_file: str = "logs/train.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        
        c_format = logging.Formatter("[%(levelname)s] %(message)s")
        f_format = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger

class MetricLogger:
    """
    Logs epoch-level metrics into a CSV file for plotting loss & accuracy curves.
    """
    def __init__(self, csv_file: str = "logs/metrics.csv"):
        self.csv_file = csv_file
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        if not os.path.exists(csv_file):
            with open(csv_file, "w", encoding="utf-8") as f:
                f.write("epoch,stage,loss,mAP50,precision,recall,f1_score\n")

    def log(self, epoch: int, stage: str, metrics: Dict[str, float]):
        loss = metrics.get("loss", 0.0)
        map50 = metrics.get("mAP50", 0.0)
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        
        with open(self.csv_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{stage},{loss:.4f},{map50:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n")
