import torchvision.models as models #import mobilenet_v2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch
from collections import OrderedDict

class ModelMobilenetV2():
  @classmethod
  def get_model(cls, num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)

    #model.roi_heads.mask_predictor = None

    return model

class ModelMobilenetV2_int8():
  @classmethod
  def get_model(cls, num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = models.quantization.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)

    #model.roi_heads.mask_predictor = None

    return model

'''
class GeneralizedRCNNTransform_int8(GeneralizedRCNNTransform):
  def __init__(self, parent, quantizer):
    super(GeneralizedRCNNTransform_int8, self).__init__(parent.min_size, parent.max_size, parent.image_mean, parent.image_std, parent.size_divisible, parent.fixed_size)
    self.quant = quantizer

  def forward(self, images, targets = None):
    images_temp, targets_temp = super(GeneralizedRCNNTransform_int8, self).forward(images, targets)
    images_temp.tensors = self.quant(images_temp.tensors)
    return images_temp, targets_temp


class ModelMobilenetV2_int8(FasterRCNN):
  def __init__(self, num_classes):
    backbone = models.quantization.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    super(ModelMobilenetV2_int8, self).__init__(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    
    # QuantStub converts tensors from floating point to quantized
    self.quant = torch.quantization.QuantStub()

    # DeQuantStub converts tensors from quantized to floating point
    self.dequant = torch.quantization.DeQuantStub()

    self.transform = GeneralizedRCNNTransform_int8(self.transform, self.quant)
  
  def forward(self, images, targets=None):
    x = super(ModelMobilenetV2_int8, self).forward(images, targets)
    return self.dequant(x)


  @classmethod
  def get_model(cls, num_classes):
    model = cls(num_classes)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.prepare_qat(model)
    return model
'''