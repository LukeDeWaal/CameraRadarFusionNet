import torch.nn as nn
import torchvision

#conda install -c conda-forge pytorch-model-summary


class VGGBackbone:
    """Provides Backbone Information and Utility Functions"""

    def __init__(self, backbone:str = "vgg-max-fpn"):
        self.backbone = backbone

    def retinanet(self, *args, **kwargs):
        """Returns RetinaNet model with VGG Backbone"""
        return vgg_retinanet(*args, backbone=self.backbone, **kwargs)


def vgg_retinanet(num_classes:int, backbone:str, input:bool=None, modifier:bool=None, distance:bool=False, cfg=None, **kwargs):
    """Constructs a Retinanet Model with a VGG Backbone

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
    Returns
        RetinaNet model with a VGG backbone.
    """

    # TODO: include_top functionality dropped in pytorchvision
    # include_top: whether to include the 3 fully-connected layers at the top of the network.
    # in vggmax
    # if include_top:
    #         # Classification block
    #         x = layers.Flatten(name='flatten')(x)
    #         x = layers.Dense(4096, activation='relu', name='fc1')(x)
    #         x = layers.Dense(4096, activation='relu', name='fc2')(x)
    #         x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    if backbone == "vgg16":
        vgg = torchvision.models.vgg16(num_classes=num_classes, pretrained=False)
        vgg.classifier = None
    elif backbone == "vgg19":
        vgg = torchvision.models.vgg19(num_classes=num_classes, pretrained=False)
        vgg.classifier = None
    elif "vgg-max" in backbone:
        vgg = vggmax.custom()
    else:
        raise ValueError("Backbone '{}' not supported.".format(backbone))
    print(vgg)



if __name__ == "__main__":
    a = VGGBackbone("vgg16").retinanet(3)