import wandb
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
    with wandb.init() as run:
        config = wandb.config
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("my_dataset_train",)
        cfg.DATASETS.TEST = ("my_dataset_val",)
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = config.batch_size
        cfg.SOLVER.BASE_LR = config.learning_rate
        cfg.SOLVER.WARMUP_ITERS = 1000
        cfg.SOLVER.MAX_ITER = config.max_iter
        cfg.SOLVER.STEPS = (1000, 1500)
        cfg.SOLVER.GAMMA = 0.05
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Set your number of classes
        cfg.TEST.EVAL_PERIOD = 500

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(cfg, "my_dataset_test")
        inference = inference_on_dataset(trainer.model, val_loader, evaluator)
        ap50 = inference['bbox']['AP50']
        print(ap50)
        wandb.log({'AP50': ap50})

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'AP50',
      'goal': 'maximize'   
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'values': [2, 4, 8, 16]
        },  
        'max_iter': {
            'values': [3500, 4000, 4500]
        }
    }
}

wandb.login()  
sweep_id = wandb.sweep(sweep_config, project="iseg_sweep_03")
register_datasets()
wandb.agent(sweep_id, function=train)
