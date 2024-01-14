import os
import cv2
import numpy as np
import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.config import get_cfg

# Configuration setup
cfg = get_cfg()
cfg.merge_from_file("path/to/your/config/file.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Adjust the threshold here
predictor = DefaultPredictor(cfg)

# Load test dataset
dataset_dicts = DatasetCatalog.get("my_dataset_test")

def get_iou(pred_box, gt_box):
    # Compute IoU between two bounding boxes
    ixmin = max(pred_box[0], gt_box[0])
    iymin = max(pred_box[1], gt_box[1])
    ixmax = min(pred_box[2], gt_box[2])
    iymax = min(pred_box[3], gt_box[3])

    width = max(ixmax - ixmin, 0)
    height = max(iymax - iymin, 0)
    area_intersection = width * height

    area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    area_union = area_pred + area_gt - area_intersection

    iou = area_intersection / area_union if area_union != 0 else 0
    return iou

iou_threshold = 0.5  # IoU threshold
TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives

for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    predictions = outputs["instances"].to("cpu")
    
    gt_boxes = [obj["bbox"] for obj in d["annotations"]]
    pred_boxes = [x.numpy() for x in predictions.pred_boxes]

    matched_gt_boxes = []
    for pred_box in pred_boxes:
        match_found = False
        for gt_box in gt_boxes:
            iou = get_iou(pred_box, gt_box)
            if iou >= iou_threshold and gt_box not in matched_gt_boxes:
                match_found = True
                matched_gt_boxes.append(gt_box)
                TP += 1
                break
        if not match_found:
            FP += 1
    FN += len(gt_boxes) - len(matched_gt_boxes)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

