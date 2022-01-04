import torch
import math
import numpy as np
import json

def validate_one_epoch(model, dataset, device, mode = 'f32'):
  model.train()
  losses = []

  input_dtype = None
  if mode == 'int8':
    input_dtype = torch.qint8
  elif mode == 'f16':
    input_dtype = torch.float16

  for images, targets in dataset:
    images = list(image.to(device, dtype=input_dtype) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
    with torch.no_grad():
      val_loss_dict = model(images, targets)
      losses.append(float(sum(loss for loss in val_loss_dict.values())))

  return np.array(losses).mean()

def save_loss(loss_list, model_name):
  with open("/content/drive/MyDrive/Final Project/validation_data/"+model_name+".json", 'w') as outfile:
    json.dump(loss_list, outfile)
