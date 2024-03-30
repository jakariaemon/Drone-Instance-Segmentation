from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset  
from detectron2.data.datasets import register_coco_instances   
from detectron2.engine import  DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg 
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode 
import cv2
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
register_coco_instances("my_dataset_test", {}, "test.json", "test")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(predictor.model, val_loader, evaluator) 

my_dataset_test_metadata = MetadataCatalog.get("my_dataset_test")
# Assuming 'predictor' and 'my_dataset_test_metadata' are defined.
file_paths = ['test3.jpg', 'test4.jpg', 'test3.jpg']
subcaptions = ['Original Images', 'Bounding Box Images', 'Masked Images']

# Define a function to display a row of images
def plot_row(images, subcaption, axs):
    for idx, img in enumerate(images):
        axs[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[idx].axis('off')
    plt.suptitle(subcaption, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to fit the suptitle

# Prepare the data for each type of image
original_images = []
bbox_images = []
masked_images = []

# Process images
for file_path in file_paths:
    im = cv2.imread(file_path)
    
    # Run the predictor
    outputs = predictor(im)
    
    # Original image
    original_images.append(im)
    
    # BBox image
    v = Visualizer(im[:, :, ::-1], metadata=my_dataset_test_metadata, scale=0.5)
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu()
    for box in pred_boxes:
        v.draw_box(box, edge_color="g")
    out = v.get_output()
    bbox_images.append(out.get_image()[:, :, ::-1])
    
    # Masked image
    v = Visualizer(im[:, :, ::-1], metadata=my_dataset_test_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    masked_images.append(out.get_image()[:, :, ::-1])

fig, axs = plt.subplots(1, 3, figsize=(15, 6))  
plot_row(original_images, subcaptions[0], axs)
plt.subplots_adjust(wspace=0.1)  
plt.savefig('original_images.png', bbox_inches='tight')
plt.close()


fig, axs = plt.subplots(1, 3, figsize=(15, 6))  
plot_row(bbox_images, subcaptions[1], axs)
plt.subplots_adjust(wspace=0.1) 
plt.savefig('bbox_images.png', bbox_inches='tight')
plt.close()


fig, axs = plt.subplots(1, 3, figsize=(15, 6))  
plot_row(masked_images, subcaptions[2], axs)
plt.subplots_adjust(wspace=0.1)  
plt.savefig('masked_images.png', bbox_inches='tight')
plt.close()
