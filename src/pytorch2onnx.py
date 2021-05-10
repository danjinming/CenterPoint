from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory

if __name__ == '__main__':
  opt = opts().parse()
  thresh = 0.3
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  Detector = detector_factory[opt.task]
  
  detector = Detector(opt)

  #detector.model
  dummy_input = torch.randn(1,3,512,512).cuda()
  torch.onnx.export(detector.model, dummy_input, "PiggyPoint.onnx", keep_initializers_as_inputs=True)
