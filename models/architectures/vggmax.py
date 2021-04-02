import os
import math
import argparse

import torch

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

def pad(x: torch.Tensor, padding: str or bool, kernel_size: tuple):
    assert padding in ["same", "valid"]
    if padding == "same":
        return torch.nn.functional.pad(x, (math.ceil(kernel_size[1] / 2), math.floor(kernel_size[1] / 2),
                                           math.ceil(kernel_size[0] / 2), math.floor(kernel_size[0] / 2)))
    else:
        return x


def activation(x: torch.Tensor, activation: str or bool):
    assert activation in ["ReLU", None]
    if activation == "ReLU":
        return torch.nn.functional.relu(x)
    else:
        return x


def MinMaxPool2D(x):
    maxpool2D = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    max_x = maxpool2D(x)
    min_x = MinPool2D(x)
    return torch.cat((max_x, min_x), 3)


def MinPool2D(x, padding="valid"):
    max_val = torch.max(x) + 1
    if x.shape[2] <= 20:
        padding = "same"
    is_zero = max_val * torch.eq(x, 0)
    x = is_zero + x
    x = pad(x, padding, (2, 2))
    maxpool2D = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    min_x = maxpool2D(-x)
    is_result_zero = max_val * torch.equal(min_x, max_val)
    min_x = min_x - is_result_zero
    return min_x


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, activation: str = None):
        super(Linear, self).__init__()
        self.activation = activation
        self.Linear = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.Linear(x)
        x = activation(x, self.activation)
        return x


class Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding: str, activation: str = None, stride: tuple = None):
        super(Conv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.Conv2D = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride)

    def forward(self, x):
        x = pad(x, self.padding, self.kernel_size)
        x = self.Conv2D(x)
        x = activation(x, self.activation)
        return x


class VGGBlock(torch.nn.Module):
    def __init__(self, n_block: int, conv_layers: int, in_channels, out_channels, cfg):
        super(VGGBlock, self).__init__()
        self.cfg = cfg
        self.n_block = n_block
        self.layers = conv_layers

        # Camera
        for i in range(conv_layers):
            setattr(self, f"Conv2D_{i+1}",
                    Conv2D(in_channels=int(in_channels), out_channels=int(out_channels), stride=(2, 2),
                           kernel_size=(3, 3), padding="same", activation="ReLU"))
            in_channels = out_channels
        if self.cfg.pooling == "maxmin":
            self.Pooling = Lambda(MinMaxPool2D)
        else:
            self.Pooling = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == "min":
                self.Radar = Lambda(MinPool2D)
            elif self.cfg.pooling == 'maxmin':
                self.Radar = Lambda(MinMaxPool2D)
            elif self.cfg.pooling == 'conv':
                self.Radar = Conv2D(in_channels=self.cfg.channels, out_channels=int(64 * self.cfg.network_width),
                                    kernel_size=(3, 3), stride=(2, 2), padding="same", activation="relu")
            else:
                self.Radar = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x, y):
        for i in range(self.layers):
            layer = getattr(self, f"Conv2D_{i+1}")
            x = layer(x)
        x = self.Pooling(x)
        y = self.Radar(y)

        # Concatenate Radar to Camera
        if self.n_block in self.cfg.fusion_blocks:
            x = torch.cat((x, y), 3)
        return x, y


class VGGmax(torch.nn.Module):
    def __init__(self,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 cfg=None,
                 **kwargs):

        """Instantiates the VGG16 architecture.
            Optionally loads weights pre-trained on ImageNet.
            # Arguments
                include_top: whether to include the 3 fully-connected
                    layers at the top of the network.
                weights: one of `None` (random initialization),
                      'imagenet' (pre-training on ImageNet),
                      or the path to the weights file to be loaded.
                input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
                input_shape: optional shape tuple, only to be specified
                    if `include_top` is False (otherwise the input shape
                    has to be `(224, 224, 3)`
                    (with `channels_last` data format)
                    or `(3, 224, 224)` (with `channels_first` data format).
                    It should have exactly 3 input channels,
                    and width and height should be no smaller than 32.
                    E.g. `(200, 200, 3)` would be one valid value.
                pooling: Optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                        the 4D tensor output of the
                        last convolutional layer.
                    - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                        be applied.
                classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
                fusion_blocks: list of indexes giving the blocks where radar and image is
                    concatenated. Input Layer is targeted by 0.
            # Returns
                A Pytorch Model
            # Raises
                ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
            """

        # Read config variables
        self.cfg = cfg
        self.fusion_blocks = cfg.fusion_blocks

        if not (weights in {"imagenet", None} or os.path.exists(weights)):
            raise ValueError("Incorrect Weight Initialization, "
                             "the weights should be either None (random initialization),"
                             "imagenet (pretrained on imagenet),"
                             "or a path to the weights should be given")

        if weights == "imagenet" and include_top and classes != 1000:
            raise ValueError("If using weights pretrained on imagenet with include_top = True then classes = 1000")

        super(VGGmax, self).__init__()
        # Assuming that length of cfg.channels are the input channels of VGG model
        self.VGGBlock_1 = VGGBlock(n_block=1, conv_layers=2, in_channels=len(self.cfg.channels),
                                   out_channels=64*self.cfg.network_width, cfg=cfg)
        self.VGGBlock_2 = VGGBlock(n_block=2, conv_layers=2, in_channels=64*self.cfg.network_width,
                                   out_channels=128*self.cfg.network_width, cfg=cfg)
        self.VGGBlock_3 = VGGBlock(n_block=3, conv_layers=3, in_channels=128*self.cfg.network_width,
                                   out_channels=256*self.cfg.network_width, cfg=cfg)
        self.VGGBlock_4 = VGGBlock(n_block=4, conv_layers=3, in_channels=256*self.cfg.network_width,
                                   out_channels=512*self.cfg.network_width, cfg=cfg)
        self.VGGBlock_5 = VGGBlock(n_block=5, conv_layers=3, in_channels=512*self.cfg.network_width,
                                   out_channels=512*self.cfg.network_width, cfg=cfg)

        # if include_top:
        #     self.Dense_1 = Linear(4096)

    def forward(self, x):
        # Block 0 - Fusion
        if len(self.cfg.channels) > 3:
            image_input = x[:, :3, :, :]
            radar_input = x[:, 3:, :, :]
            if 0 in self.fusion_blocks:
                x = torch.cat((image_input, radar_input), 1)
            else:
                x = image_input

        x, y = self.VGGBlock_1(x, radar_input)
        x, y = self.VGGBlock_2(x, y)
        x, y = self.VGGBlock_3(x, y)
        x, y = self.VGGBlock_4(x, y)
        x, y = self.VGGBlock_5(x, y)

        if self.include_top:
            x = torch.flatten(x)
            print(x.shape)


        return None

