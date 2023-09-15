#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from model.pips import Pips

def build_sam(configs):
  from mobile_encoder.setup_mobile_sam import setup_model
  from segment_anything import SamPredictor

  num_gpu = configs['num_gpu']
  use_gpu = (len(num_gpu) > 0)
  mobile_sam = setup_model()
  checkpoint = torch.load(configs['sam_model_path'])
  mobile_sam.load_state_dict(checkpoint,strict=True)

  if use_gpu:
    mobile_sam = mobile_sam.to(torch.device('cuda:{}'.format(num_gpu[0])))
  mobile_sam.eval()
  predictor = SamPredictor(mobile_sam)
  return predictor

def build_pips(configs):
  pips = Pips(stride=4).cuda()
  checkpoint = torch.load(configs['pips_model_path'])
  pips.load_state_dict(checkpoint['model_state_dict'], strict=False)
  pips.eval()
  return pips