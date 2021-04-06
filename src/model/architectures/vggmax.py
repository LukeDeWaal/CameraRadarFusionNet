import os
import math
import re
import functools

import torch

WEIGHTS_PATH = 'https://download.pytorch.org/models/vgg16-397923af.pth'
WEIGHTS_PATH_BN = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def find_number(text, c):
    return re.findall(r'%s(\d+)' % c, text)[0]


def get_n(block, layer, layers: list):
    n = 0
    for i in range(block-1):
        n += 2 * (layers[i] - 1) + 3
    n += 2 * (layer - 1)
    return n


def output_size(x, blocks):
    if isinstance(x, torch.Tensor):
        height, width = x.shape[-2], x.shape[-1]
    else:
        height, width = x[-2], x[-1]
    for i in range(blocks):
        height, width = math.floor(height/2), math.floor(width/2)
    return height, width


def pad(x: torch.Tensor, padding: str or bool, kernel_size: tuple, stride: tuple = (1,1)):
    assert padding in ["same", "valid"]
    if padding == "same":
        if x.shape[-2] % stride[0] == 0:
            height_pad = max(kernel_size[0] - stride[0], 0)
        else:
            height_pad = max(kernel_size[0] - (x.shape[-2] % stride[0]), 0)
        if x.shape[-1] % stride[1] == 0:
            width_pad = max(kernel_size[1] - stride[1], 0)
        else:
            width_pad = max(kernel_size[1] - (x.shape[-1] % stride[1]), 0)

        pad_top = height_pad // 2
        pad_bottom = height_pad - pad_top
        pad_left = width_pad // 2
        pad_right = width_pad - pad_left

        return torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    else:
        return x


def activation(x: torch.Tensor, activation: str or bool):
    assert activation in ["relu", "softmax", None]
    if activation == "relu":
        return torch.nn.functional.relu(x)
    elif activation == "softmax":
        return torch.nn.functional.softmax(x, dim=1)
    else:
        return x


def MinMaxPool2D(x):
    stride = (2, 2)
    kernel_size = (2, 2)
    maxpool2D = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    max_x = maxpool2D(x)
    min_x = MinPool2D(x, kernel_size=kernel_size, stride=stride)
    return torch.cat((max_x, min_x), 3)


def MinPool2D(x, kernel_size, stride, padding="valid"):
    max_val = torch.max(x) + 1
    if x.shape[2] <= 20:
        padding = "same"
    is_zero = max_val * torch.eq(x, 0)
    x = is_zero + x
    x = pad(x, padding, kernel_size=kernel_size, stride=stride)
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
    def __init__(self, in_channels, out_channels, kernel_size, padding: str, activation: str or bool = None, stride: tuple = (1, 1)):
        super(Conv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.stride = stride
        self.Conv2D = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride)

    def forward(self, x):
        x = pad(x, self.padding, self.kernel_size, self.stride)
        x = self.Conv2D(x)
        x = activation(x, self.activation)
        return x


class VGGBlock(torch.nn.Module):
    def __init__(self, n: int, conv_layers: int, in_channels, out_channels, cfg):
        super(VGGBlock, self).__init__()
        self.cfg = cfg
        self.n = n
        self.layers = conv_layers

        if self.n in self.cfg.fusion_blocks:
            in_channels += 2

        # Camera
        for i in range(conv_layers):
            setattr(self, f"layer_{i+1}",
                    Conv2D(in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=(3, 3),
                           padding="same", activation="relu"))
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
            layer = getattr(self, f"layer_{i+1}")
            x = layer(x)
        x = self.Pooling(x)
        y = self.Radar(y)
        # Concatenate Radar to Camera
        if self.n+1 in self.cfg.fusion_blocks:
            x = torch.cat((x, y), 1)
        return x, y


class VGGmax(torch.nn.Module):
    def __init__(self,
                 include_top=True,
                 weights='imagenet',
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
                weights: one of `None` (random initialization) or
                      'imagenet' (pre-training on ImageNet)
                input_shape: shape tuple, to be specified if `include_top`
                    is True E.g. `(5, 224, 224)` with (C, W, H)
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
        self.include_top = include_top
        self.input_shape = input_shape
        self.classes = classes
        self.pooling = pooling

        self.blocks = 5
        self.conv_layers = [2, 2, 3, 3, 3]
        self.channels = [3, 64, 128, 256, 512, 512]


        if not (weights in {"imagenet", None} or os.path.exists(weights)):
            raise ValueError("Incorrect Weight Initialization, "
                             "the weights should be either None (random initialization),"
                             "imagenet (pretrained on imagenet),"
                             "or a path to the weights should be given")

        if weights == "imagenet" and include_top and classes != 1000:
            raise ValueError("If using weights pretrained on imagenet with include_top = True then classes = 1000")

        super(VGGmax, self).__init__()
        # Assuming that length of cfg.channels are the input channels of VGG model
        for i in range(self.blocks):
            setattr(self, f"block_{i+1}",
                    VGGBlock(n=i, conv_layers=self.conv_layers[i],
                             in_channels=self.channels[i]*self.cfg.network_width,
                             out_channels=self.channels[i+1]*self.cfg.network_width, cfg=cfg))
        if self.input_shape and self.include_top:
            height, width = output_size(input_shape, self.blocks)
            self.classifier_in_features = height*width*(self.channels[-1]+2) if self.blocks in self.cfg.fusion_blocks \
                else height*width*self.channels[-1]
            self.classifier_1 = Linear(in_features=self.classifier_in_features, out_features=4096, activation="relu")
            self.classifier_2 = Linear(in_features=4096, out_features=4096, activation="relu")
            self.classifier_3 = Linear(in_features=4096, out_features=classes, activation="softmax")

        if weights == 'imagenet':
            pretrained_state = torch.hub.load_state_dict_from_url(WEIGHTS_PATH)
            model_state = self.state_dict()
            for model_key in model_state.keys():
                if "block" in model_key:
                    block, layer = int(find_number(model_key, "block_")), int(find_number(model_key, "layer_"))
                    n = get_n(block, layer, self.conv_layers)
                    pretrained_key = f"features.{n}." + model_key.split(".")[-1]
                    pretrained_data = pretrained_state[pretrained_key]
                    if block-1 in self.fusion_blocks and layer == 1 and model_key.split(".")[-1] == "weight":
                        layer_data = rgetattr(self, model_key+'.data')
                        pretrained_data = torch.cat((pretrained_state[pretrained_key], layer_data[:, -2:, :, :]), 1)
                    rsetattr(self, model_key+'.data', pretrained_data)
                if include_top and "classifier" in model_key:
                    c = 3*(int(find_number(model_key, "classifier_"))-1)
                    classifier_key = f"classifier.{c}." + model_key.split(".")[-1]
                    classifier_data = pretrained_state[classifier_key]
                    if self.blocks in self.fusion_blocks and model_key.split(".")[-1] == "weight" and c == 0:
                        layer_data = rgetattr(self, model_key + '.data')
                        p = self.classifier_in_features - pretrained_state[classifier_key].shape[1]
                        classifier_data = torch.cat((pretrained_state[classifier_key], layer_data[:, -p:]), 1)
                    rsetattr(self, model_key+'.data', classifier_data)

    def forward(self, x):
        # Block 0 - Fusion
        if len(self.cfg.channels) > 3:
            image_input = x[:, :3, :, :]
            radar_input = x[:, 3:, :, :]
            if 0 in self.fusion_blocks:
                x = torch.cat((image_input, radar_input), 1)
            else:
                x = image_input

        x, y = self.block_1(x, radar_input)
        x, y = self.block_2(x, y)
        x, y = self.block_3(x, y)
        x, y = self.block_4(x, y)
        x, y = self.block_5(x, y)

        if self.include_top:
            x = torch.flatten(x, start_dim=1, end_dim=3)
            x = self.classifier_1(x)
            x = self.classifier_2(x)
            x = self.classifier_3(x)
        else:
            if self.pooling == "avg":
                x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
            if self.pooling == "max":
                x = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)
        return x
