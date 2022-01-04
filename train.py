import numpy as np
import torch

import utils
from validation import validate_one_epoch, save_loss
from functions import get_transform, load_model, train_one_epoch, save_model, convert_model_backbone
from ColaBeerDataset import ColaBeerDataset

#import models
from ModelResnet50 import ModelResnet50
from ModelMobilenetV2 import ModelMobilenetV2, ModelMobilenetV2_int8
from ModelMobilenetV3 import ModelMobileNetV3, ModelMobileNetV3L, ModelMobileNetV3L_int8
from ModelSSD import ModelSSDlite320

path_train = 'video1/train/'
path_test = 'video1/test/'

dataset = ColaBeerDataset(path_train,get_transform(train=True))
dataset_test = ColaBeerDataset(path_test, get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1337)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices)
indices_test = torch.randperm(len(dataset_test)).tolist()
dataset_test = torch.utils.data.Subset(dataset_test, indices_test)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

drop_last = True
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn, drop_last=drop_last)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# background and 2 types of cans
num_classes = 3

mode = 'f32'
use_int8 = True

#model_type = ModelResnet50
#model_type = ModelMobilenetV2
model_type = ModelMobilenetV2_int8
#model_type = ModelMobileNetV3
#model_type = ModelMobileNetV3L
#model_type = ModelMobileNetV3L_int8
#model_type = ModelSSDlite320

#model_name = "bbox_detector_resnet_e30_lr-4_reg-4_g2_adam"
#model_name = "bbox_detector_resnet_half_e30_lr-4_g2_adam"
#model_name = "bbox_detector_mobilenetv2_e30_lr-4_g2_adam"
#model_name = "bbox_detector_mobilenetv2_half_e30_lr-4_g2_adam"
model_name = "bbox_detector_mobilenetv2_int8_e30_lr-4_g2_adam_converted"
#model_name = "bbox_detector_mobilenetv3_e30_lr-4_g2_adam"
#model_name = "bbox_detector_mobilenetv3_half_e30_lr-4_g2_adam"
#model_name = "bbox_detector_mobilenetv3L_e30_lr-4_g2_adam"
#model_name = "bbox_detector_mobilenetv3L_half_e30_lr-4_g2_adam"
#model_name = "bbox_detector_mobilenetv3L_int8_e30_lr-4_g2_adam_converted"
#model_name = "bbox_detector_SSDlite320_e30_lr-3_reg-3_g2_adam"
#model_name = "bbox_detector_SSDlite320_half_e20_lr-3_reg-3_g2_adam"

# get the model using our helper function
model = model_type.get_model(num_classes)

# move model to the right device
model.to(device)

if mode == 'f16':
  model.half()
  for layer in model.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
      layer.float()

if mode == 'f16':
  eps = 1e-4
else:
  eps = 1e-8

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=1e-6)
optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=1e-6, eps=eps)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
scaler = torch.cuda.amp.GradScaler()


num_epochs = 30
losses_val = []
losses_train = []
for epoch in range(num_epochs):
    # train for one epoch, printing every 100 iterations
    loss_train = train_one_epoch(model, optimizer, data_loader, device, epoch, 100, mode, scaler)

    # update the learning rate
    lr_scheduler.step()
    
    # evaluate on the test dataset
    if use_int8:
      model_int8 = convert_model_backbone(model)
      loss_val = validate_one_epoch(model_int8, data_loader_test, torch.device('cpu'))
    else:
      loss_val = validate_one_epoch(model, data_loader_test, device, mode)

    print(f'Epoch: [{epoch}] Loss training: {loss_train:.5f}, Loss validation: {loss_val:.5f}')
    losses_train.append(loss_train)
    losses_val.append(loss_val)

print("Finished.")


#Saving model
save_loss(losses_val, model_name+'_val')
save_loss(losses_train, model_name+'_train')

if mode == 'amp':
  save_model(f"{model_name}_amp", model, optimizer, scaler)
elif use_int8:
  save_model(model_name, model_int8)
else:
  save_model(model_name, model)
