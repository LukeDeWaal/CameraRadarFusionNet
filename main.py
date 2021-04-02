import os
import argparse
import torch

from models.utils.config import get_config
from models.architectures.vggmax import VGGmax




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

    x = torch.randn(3, 5, 200, 200)
    model = VGGmax(cfg=cfg)
    print(model)
    model(x)