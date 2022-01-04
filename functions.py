import sys
sys.path.append('/content/drive/MyDrive/Final Project')
import transforms as T
import utils
import torch
import math
import sys
import copy

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    #transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, mode = 'f32', scaler = None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    if mode == 'amp':
        use_amp = True
    else:
        use_amp = False

    input_dtype = None
    if mode == 'int8':
        input_dtype = torch.qint8
    elif mode == 'f16':
        input_dtype = torch.float16
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        optimizer.zero_grad()

        images = list(image.to(device, dtype=input_dtype) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        if use_amp and scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return float(metric_logger.loss.total / metric_logger.loss.count)

def save_model(name, model, optimizer = None, scaler = None):
    torch.save(model.state_dict(), f'/content/drive/MyDrive/Final Project/models/{name}.pth')
    if optimizer is not None and scaler is not None:
        torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, f'models/{name}_opt_scl.pth')

def load_model(name, model, optimizer = None, scaler = None):
    #dev = torch.cuda.current_device()
    device = None if torch.cuda.is_available() else torch.device('cpu')
    #model.load_state_dict(torch.load(f'/content/drive/MyDrive/Final Project/models/{name}.pth', map_location = lambda storage, loc: storage.cuda(dev)), False)
    model.load_state_dict(torch.load(f'/content/drive/MyDrive/Final Project/models/{name}.pth', map_location = device), False)
    if optimizer is not None and scaler is not None:
        #checkpoint = torch.load(f'/content/drive/MyDrive/Final Project/models/{name}_opt_scl.pth', map_location = lambda storage, loc: storage.cuda(dev))
        checkpoint = torch.load(f'/content/drive/MyDrive/Final Project/models/{name}_opt_scl.pth', map_location = device)
        optimizer.load_state_dict(checkpoint["optimizer"], False)
        scaler.load_state_dict(checkpoint["scaler"], False)

def convert_model_backbone(model):
    model_int8 = copy.deepcopy(model)
    model_int8.cpu()
    model_int8.backbone = torch.quantization.convert(model_int8.backbone)
    return model_int8

