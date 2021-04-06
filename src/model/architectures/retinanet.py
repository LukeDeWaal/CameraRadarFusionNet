# import tensorflow.keras as keras
# from crfnet.model import initializers
# from .. import layers
# from . import assert_training_model

import torch
import math

try:
    from utils import AnchorParameters, Conv2D, output_size
except (ModuleNotFoundError, ImportError):
    from src.model.architectures.utils import AnchorParameters, Conv2D, output_size


def build_pyramid(models, features):
    """ Applies all submodels to each FPN level.
    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.
    Returns
        A list of tensors, one for each submodel.
    """
    x = []
    for model in models:
        output = tuple(model(f) for f in features)
        output = torch.cat(output, dim=0)
        x.append(output)
    return x


class Classification_Model(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 num_anchors,
                 pyramid_feature_size=256,
                 prior_probability=0.01,
                 classification_feature_size=256):
        """ Creates the default regression submodel.
        Args
            num_classes                 : Number of classes to predict a score for at each feature level.
            num_anchors                 : Number of anchors to predict classification scores for at each feature level.
            pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
            classification_feature_size : The number of filters to use in the layers in the classification submodel.
        Returns
            A keras.models.Model that predicts classes for each anchor.
        """
        super(Classification_Model, self).__init__()
        self.pyramid_feature_size = pyramid_feature_size
        self.probability = prior_probability
        self.num_classes = num_classes

        self.features = torch.nn.Sequential()

        for i in range(4):
            self.features.add_module(f"'pyramid_classification_{i}", Conv2D(in_channels=self.pyramid_feature_size,
                                                                            out_channels=classification_feature_size,
                                                                            kernel_size=3, stride=1,
                                                                            padding='same', activation='relu'))
            torch.nn.init.normal_(self.features[i].Conv2D.weight.data, mean=0.0, std=0.01)
            torch.nn.init.constant_(self.features[i].Conv2D.bias.data, val=0.0)
        self.features.add_module(f"'pyramid_classification_{i + 1}", Conv2D(in_channels=classification_feature_size,
                                                                            out_channels=num_anchors * num_classes,
                                                                            kernel_size=3, stride=1,
                                                                            padding='same', activation='relu'))
        torch.nn.init.normal_(self.features[i].Conv2D.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.features[i].Conv2D.bias.data, val=-math.log((1-self.probability)/self.probability))
        self.features.add_module("sigmoid", torch.nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (-1, self.num_classes))
        return x


class Regression_Model(torch.nn.Module):
    def __init__(self,
                 num_values,
                 num_anchors,
                 pyramid_feature_size=256,
                 regression_feature_size=256):
        """ Creates the default regression submodel.
        Args
            num_values              : Number of values to regress.
            num_anchors             : Number of anchors to regress for each feature level.
            pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
            regression_feature_size : The number of filters to use in the layers in the regression submodel.
            name                    : The name of the submodel.
        Returns
            A keras.models.Model that predicts regression values for each anchor.
        """
        super(Regression_Model, self).__init__()
        self.pyramid_feature_size = pyramid_feature_size
        self.num_values = num_values
        self.features = torch.nn.Sequential()

        for i in range(4):
            self.features.add_module(f"pyramid_regression_{i}", Conv2D(in_channels=self.pyramid_feature_size,
                                                                       out_channels=regression_feature_size,
                                                                       kernel_size=3, stride=1,
                                                                       padding='same', activation='relu'))
            torch.nn.init.normal_(self.features[i].Conv2D.weight.data, mean=0.0, std=0.01)
            torch.nn.init.constant_(self.features[i].Conv2D.bias.data, val=0.0)

        self.features.add_module(f"pyramid_regression_{i + 1}", Conv2D(in_channels=self.pyramid_feature_size,
                                                                       out_channels=num_anchors * self.num_values,
                                                                       kernel_size=3,
                                                                       stride=1, padding='same', activation='relu'))
        torch.nn.init.normal_(self.features[i + 1].Conv2D.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.features[i + 1].Conv2D.bias.data, val=0.0)

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (-1, self.num_values))
        return x


def default_submodels(num_classes, num_anchors, distance=False):
    """ Create a list of default submodels used for object detection.
    The default submodels contains a regression submodel and a classification submodel.
    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.
    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    if distance:
        return torch.nn.ModuleList([
            Regression_Model(4, num_anchors),
            Classification_Model(num_classes, num_anchors),
            Regression_Model(1, num_anchors)
        ])
    else:
        return torch.nn.ModuleList([
            Regression_Model(4, num_anchors),
            Classification_Model(num_classes, num_anchors),
        ])


class Create_Pyramid_Features(torch.nn.Module):
    def __init__(self, feature_size=256, radar=True, cfg=None):
        """ Creates the FPN layers on top of the backbone features.
        Args
            feature_size : The feature size to use for the resulting feature levels.
        Returns
            A list of feature levels [P3, P4, P5, P6, P7].
        """
        super(Create_Pyramid_Features, self).__init__()
        self.cfg = cfg
        self.feature_size = feature_size
        self.vgg_channels = cfg.vgg_channels

        if radar:
            self.feature_size -= 2

        for i in range(len(self.vgg_channels)):
            if i in self.cfg.fusion_blocks:
                self.vgg_channels[i] += 2

        self.output_sizes = [output_size(self.cfg.image_size, i) for i in range(3, 6)]

        self.P5_layers = torch.nn.ModuleList([
            Conv2D(in_channels=self.vgg_channels[-1], out_channels=self.feature_size, kernel_size=1, stride=1,
                   padding="same"),
            torch.nn.Upsample(size=self.output_sizes[-2], mode='nearest'),
            Conv2D(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1,
                   padding="same"),
        ])
        self.P4_layers = torch.nn.ModuleList([
            Conv2D(in_channels=self.vgg_channels[-2], out_channels=self.feature_size, kernel_size=1, stride=1,
                   padding="same"),
            torch.nn.Upsample(size=self.output_sizes[-3], mode='nearest'),
            Conv2D(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1,
                   padding="same")
        ])

        self.P3_layers = torch.nn.ModuleList([
            Conv2D(in_channels=self.vgg_channels[-3], out_channels=self.feature_size, kernel_size=1, stride=1,
                   padding="same"),
            Conv2D(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=1,
                   padding="same")
        ])

        self.P6_layers = Conv2D(in_channels=self.vgg_channels[-1], out_channels=self.feature_size, kernel_size=3,
                                stride=2,
                                padding='same')

        self.P7_layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            Conv2D(in_channels=self.feature_size, out_channels=self.feature_size, kernel_size=3, stride=2,
                   padding='same')
        )

    def forward(self, backbone_layers, radar_layers=None):
        C3, C4, C5 = backbone_layers
        P5 = self.P5_layers[0](C5)
        P5_upsampled = self.P5_layers[1](P5)
        P5 = self.P5_layers[2](P5)

        P4 = self.P4_layers[0](C4)
        P4 = torch.add(P4, P5_upsampled)
        P4_upsampled = self.P4_layers[1](P4)
        P4 = self.P4_layers[2](P4)

        P3 = self.P3_layers[0](C3)
        P3 = torch.add(P3, P4_upsampled)
        P3 = self.P3_layers[1](P3)

        P6 = self.P6_layers(C5)

        P7 = self.P7_layers(P6)

        if radar_layers:
            R3 = radar_layers[2]
            R4 = radar_layers[3]
            R5 = radar_layers[4]
            R6 = radar_layers[5]
            R7 = radar_layers[6]
            P3 = torch.cat((P3, R3), dim=1)
            P4 = torch.cat((P4, R4), dim=1)
            P5 = torch.cat((P5, R5), dim=1)
            P6 = torch.cat((P6, R6), dim=1)
            P7 = torch.cat((P7, R7), dim=1)
        return [P3, P4, P5, P6, P7]


class Retinanet(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 num_anchors=None,
                 submodels=None,
                 distance=False,
                 cfg=None):
        """ Construct a RetinaNet model on top of a backbone.
        This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).
        Args
            inputs                  : keras.layers.Input (or list of) for the input to the model.
            num_classes             : Number of classes to classify.
            num_anchors             : Number of base anchors.
            create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
            submodels               : Submodels to run on each feature map (default is regression and classification submodels).
            name                    : Name of the model.
        Returns
            A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.
            The order of the outputs is as defined in submodels:
            ```
            [
                regression, classification, other[0], other[1], ...
            ]
            ```
        """
        super(Retinanet, self).__init__()
        self.cfg = cfg
        self.submodels = submodels
        self.pyramid_features = Create_Pyramid_Features(cfg=self.cfg)

        if num_anchors is None:
            num_anchors = AnchorParameters.default.num_anchors()
        if self.submodels is None:
            self.submodels = default_submodels(num_classes, num_anchors, distance)


    def forward(self, backbone_outputs, radar_outputs):
        # compute pyramid features as per https://arxiv.org/abs/1708.02002
        features = self.pyramid_features(backbone_layers=backbone_outputs, radar_layers=radar_outputs)
        # for all pyramid levels, run available submodels
        pyramids = build_pyramid(self.submodels, features)
        return pyramids
