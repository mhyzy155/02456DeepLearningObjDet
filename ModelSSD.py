import torchvision.models as models #import mobilenet_v2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import ssd300_vgg16

class ModelSSDlite320():
    @classmethod
    def get_model(cls, num_classes, quantize = False):
        model = ssdlite320_mobilenet_v3_large(pretrained=True, pretrained_backbone=True)
        model.backbone.num_classes = num_classes
        return model

class ModelSSD300():
    @classmethod
    def get_model(cls, num_classes, quantize = False):
        model = ssd300_vgg16(pretrained=False, num_classes = num_classes, pretrained_backbone=True)
        return model
