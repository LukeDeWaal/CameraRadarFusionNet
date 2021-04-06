import os
import argparse
import torch

from models.architectures.vgg import VGG_Retinanet
from src.config import get_config


if __name__ == "__main__":
    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(FILE_DIRECTORY, "configs/default.cfg"))
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("ERROR: Config file \"%s\" not found" % (args.config))
    else:
        cfg = get_config(args.config)

    model_name = args.config.split('/')[-1]
    model_name = model_name.split('.')[0]
    cfg.model_name = cfg.runtime + "_" + model_name

    assert cfg.inference is False, "You are running a training in inference mode. Please check your config!"

    backbone = cfg.network

    x = torch.randn(10, 5, 240, 240)
    model = VGG_Retinanet(num_classes=8, backbone=backbone, weights=None, include_top=False, cfg=cfg)
    output = model(x)