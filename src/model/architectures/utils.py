import torch
import math


def pad(x: torch.Tensor, padding: str or bool, kernel_size: tuple, stride: tuple = (1, 1)):
    assert padding in ["same", "valid", None]
    if padding == "same":
        if isinstance(kernel_size, tuple) and isinstance(stride, tuple):
            if x.shape[-2] % stride[0] == 0:
                height_pad = max(kernel_size[0] - stride[0], 0)
            else:
                height_pad = max(kernel_size[0] - (x.shape[-2] % stride[0]), 0)
            if x.shape[-1] % stride[1] == 0:
                width_pad = max(kernel_size[1] - stride[1], 0)
            else:
                width_pad = max(kernel_size[1] - (x.shape[-1] % stride[1]), 0)
        elif isinstance(kernel_size, int) and isinstance(stride, int):
            if x.shape[-2] % stride == 0:
                height_pad = max(kernel_size - stride, 0)
            else:
                height_pad = max(kernel_size - (x.shape[-2] % stride), 0)
            if x.shape[-1] % stride == 0:
                width_pad = max(kernel_size - stride, 0)
            else:
                width_pad = max(kernel_size - (x.shape[-1] % stride), 0)
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


class Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding: str, activation: str or bool = None,
                 stride: tuple or int = (1, 1)):
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


def output_size(x, blocks):
    if isinstance(x, torch.Tensor):
        height, width = x.shape[-2], x.shape[-1]
    else:
        height, width = x[-2], x[-1]
    for i in range(blocks):
        height, width = math.floor(height / 2), math.floor(width / 2)
    return height, width


"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Original Source: https://github.com/fizyr/keras-retinanet
"""


class AnchorParameters:
    """ The parameteres that define how anchors are generated.
    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """

    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=torch.Tensor([0.5, 1, 2]),
    scales=torch.Tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]),
)

AnchorParameters.small = AnchorParameters(
    sizes=[16, 32, 64, 128, 256],
    strides=[8, 16, 32, 64, 128],
    ratios=torch.Tensor([0.5, 1, 2]),
    scales=torch.Tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]),
)
