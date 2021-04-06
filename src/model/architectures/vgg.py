import torch

try:
    from resources import model_urls
    from vggmax import VGGmax
    from retinanet import Retinanet
except (ModuleNotFoundError, ImportError):
    from .resources import model_urls
    from .vggmax import VGGmax
    from .retinanet import Retinanet


class VGGBackbone:
    """Provides Backbone Information and Utility Functions"""

    def __init__(self, backbone: str = "vgg-max-fpn"):
        self.backbone = backbone

    def retinanet(self, *args, **kwargs):
        """Returns RetinaNet model with VGG Backbone"""
        return VGG_Retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html.
        """
        if self.backbone == 'vgg16' or 'vgg-max' in self.backbone:
            resource = model_urls[self.backbone]
        elif self.backbone == 'vgg19':
            resource = model_urls[self.backbone]
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))
        return torch.hub.load_state_dict_from_url(resource)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19', 'vgg-max', 'vgg-max-fpn']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))


class VGG_Retinanet(torch.nn.Module):
    def __init__(self, num_classes, backbone='vgg-max-fpn', inputs=None, distance=False, cfg=None, **kwargs):
        """Constructs a Retinanet Model with a VGG Backbone

        Args
            num_classes: Number of classes to predict.
            backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
            inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
            modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
        Returns
            RetinaNet model with a VGG backbone.
        """

        super(VGG_Retinanet, self).__init__()
        if "vgg-max" in backbone:
            self.vgg = VGGmax(include_top=False, weights=None, cfg=cfg)
        else:
            raise ValueError("Backbone '{}' not supported.".format(backbone))

        self.retinanet = Retinanet(num_classes=num_classes, distance=distance, cfg=cfg)

    def forward(self, x):
        backbone_outputs, radar_outputs = self.vgg(x)
        outputs = self.retinanet(backbone_outputs=backbone_outputs, radar_outputs=radar_outputs)
        return outputs
