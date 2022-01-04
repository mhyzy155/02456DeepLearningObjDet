import torchvision.models as models
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN


class ModelMobileNetV3():
    @classmethod
    def get_model(cls, num_classes):
        # load a pre-trained model for classification and return
        # only the features
        backbone = models.mobilenet_v3_small(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 576

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
        #anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),
        #                            aspect_ratios=((0.5, 1.0, 2.0),))
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
                    box_roi_pool=roi_pooler,
                    rpn_pre_nms_top_n_train=20,
                    rpn_pre_nms_top_n_test=10,
                    rpn_post_nms_top_n_train=20,
                    rpn_post_nms_top_n_test=10)

        #model.roi_heads.mask_predictor = None

        return model

class ModelMobileNetV3L():
    @classmethod
    def get_model(cls, num_classes):
        backbone = models.mobilenet_v3_large(pretrained=True).features
        backbone.out_channels = 960
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
        model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    rpn_pre_nms_top_n_train=20,
                    rpn_pre_nms_top_n_test=10,
                    rpn_post_nms_top_n_train=20,
                    rpn_post_nms_top_n_test=10)

        return model

class ModelMobileNetV3L_int8():
    @classmethod
    def get_model(cls, num_classes):
        # load a pre-trained model for classification and return
        # only the features
        backbone = models.quantization.mobilenet_v3_large(pretrained=False).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 960

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
        #anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),
        #                            aspect_ratios=((0.5, 1.0, 2.0),))
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
                    box_roi_pool=roi_pooler,
                    rpn_pre_nms_top_n_train=20,
                    rpn_pre_nms_top_n_test=10,
                    rpn_post_nms_top_n_train=20,
                    rpn_post_nms_top_n_test=10)

        #model.roi_heads.mask_predictor = None

        return model
