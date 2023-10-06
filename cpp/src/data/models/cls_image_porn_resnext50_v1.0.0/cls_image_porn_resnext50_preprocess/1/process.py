#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import forge
import cv2
import numpy as np
import time



def handler(req):
  INPUT = req['rawimg'].as_ndarray()
  image = cv2.imdecode(INPUT, cv2.IMREAD_COLOR)
  resize_width = 300
  resize_height = 300

  image = cv2.resize(image,(resize_width,resize_height),interpolation=cv2.INTER_AREA)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = np.divide(image,255.)

  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])  
  mean = np.float64(mean.reshape(1, -1))
  stdinv = 1 / np.float64(std.reshape(1, -1))

  cv2.subtract(image, mean, image)  # inplace
  cv2.multiply(image, stdinv, image)  # inplace
  image = image.transpose(2, 0, 1)
  return {'preprocessed_img': image.astype('float32')}

forge.run(handler)

