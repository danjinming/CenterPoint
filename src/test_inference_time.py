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
  #prefetch_test(opt)

  thresh = 0.3
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  Detector = detector_factory[opt.task]
  
  detector = Detector(opt)

  ##detector.model
  #dummy_input = torch.randn(1,3,512,512)
  #torch.onnx.export(detector.model, dummy_input, "mv2_centernet.onnx", keep_initializers_as_inputs=True)

  img = cv2.imread("/home/lampson/2T_disk/Data/images/poseTesting.jpg")
  for i in range(0,1):
    ret = detector.run(img)
    print(ret)
    results = ret['results']
    print("-pre process time : ",ret['pre'])
    print("-net time : ",ret['net'])
    print("-decode time : ",ret['dec'])
    print("-post process time : ",ret['post'])
    for cls_id in range(1,81):
        person_results = results.get(cls_id)
        
        for p_box in person_results:
          if(p_box[4] > thresh):
              cv2.rectangle(img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0,255,0), 2, 8 ,0)

    cv2.imshow("t", img)
    cv2.waitKey(0)
