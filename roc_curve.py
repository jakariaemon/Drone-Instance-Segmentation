import numpy as np
import cv2
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import product 
from detectron2.data.datasets import register_coco_instances   
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # one class for drone, one for background
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  
predictor = DefaultPredictor(cfg)

register_coco_instances("my_dataset_test", {}, "valid/_annotations.coco.json", "valid")
my_dataset_test_metadata = MetadataCatalog.get("my_dataset_test")
dataset_dicts = DatasetCatalog.get("my_dataset_test") 
 
def get_true_label(d):
    if not d['annotations']:
        return 0
    return d['annotations'][0]['category_id']
def get_predicted_label(outputs):
    return 1 if len(outputs["instances"]) > 0 else 0  

scores = []
true_labels = []

for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    
    true_label = get_true_label(d)
    true_labels.append(true_label)
    if len(outputs["instances"]) > 0:
        score = outputs["instances"].scores[0].item()  
        scores.append(score)
    else:
        scores.append(0) 

fpr, tpr, thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Valid)')
plt.legend(loc="lower right")
plt.savefig('MaskRCNN_ROC_Curve_validation_set.png')
plt.show()
