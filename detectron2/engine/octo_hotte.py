# -*- coding: utf-8 -*-

"""
Modified from .defaults.py
"""
import os

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator, COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

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