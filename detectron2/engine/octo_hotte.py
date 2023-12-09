# -*- coding: utf-8 -*-

"""
Modified from .defaults.py
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator, COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.modeling import build_model

# Custom evaluator function 
# https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
# https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
from .LossEvalHook import LossEvalHook


def build_octo_train_aug(cfg):
    augs = []

    augs.append(T.RandomContrast(intensity_min=0.7, intensity_max=1.3))
    augs.append(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5))
    augs.append(T.RandomSaturation(intensity_min=0.75, intensity_max=1.25))
    augs.append(T.RandomFlip())
    return augs


class OctoTrainer(DefaultTrainer):
    """
    Custom trainer implemented by Horst O. - October 2022 
    
    
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. 
    """

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_octo_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)
    

    # Add an evaluator method with "LossEvalHook"
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                cfg = self.cfg,
                dataset_name = self.cfg.DATASETS.TEST[0],
                mapper = DatasetMapper(self.cfg, is_train=True),
            )))
        return hooks
    
### PREDICTION 
class OctoPredictor:
    """
    Modified from defaults.py DefaultPredictor
    
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Take one input image and produce a single output, instead of a batch.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            try:
                height, width = original_image.shape[:2]
            except AttributeError:
                return None
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions['instances'].to("cpu") # to cpu is important unless you want to overflow GPU mem


class OctoPredictorBatch:
    '''
    Batch prediction version of OctoPredictor (see above)
    Horst: On my 3070Ti this did lead to only tiny speed improvements (10 perc or less).
    Either I am doing things wrong, or the memory / processing capability 
    are not high enough to see a measurable difference with the batch sizes that 
    I am requesting.    
    
    '''
    def __init__(self, cfg, batch_size, workers=0):
        
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.batch_size = batch_size
        self.workers = workers
        assert self.workers > -1, f'Number of workers cannot be negative. Choose 0 to disable parallel processing'
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __collate(self, batch):
        data = []
        for image in batch:
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]

            image = self.aug.get_transform(image).apply_image(image)
            image = image.astype("float32").transpose(2, 0, 1)
            image = torch.as_tensor(image)
            data.append({"image": image, "height": height, "width": width})
        return data

    def __call__(self, images):
        """
        """
        loader = DataLoader(
            images,
            self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.__collate,
            pin_memory=True
        )
        with torch.no_grad():
            results = []
            for batch in loader:
                predictions = self.model(batch)
                predictions = [pred['instances'].to("cpu") for pred in predictions]
                results.append(predictions)
            return results