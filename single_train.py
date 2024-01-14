import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

def register_datasets():
    register_coco_instances("my_dataset_train", {}, "train/_annotations.coco.json", "train")
    register_coco_instances("my_dataset_val", {}, "valid/_annotations.coco.json", "valid")
    register_coco_instances("my_dataset_test", {}, "test/_annotations.coco.json", "test")

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def train():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00306982
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.STEPS = (30000, 45500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.EVAL_PERIOD = 500

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def test():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("output", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    cfg.DATASETS.TEST = ("my_dataset_test",)
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset_test")
    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(inference)

# Register datasets
register_datasets()

# Train the model
train()

# Test the model
test()
