# CRF-Net for Object Detection (Camera and Radar Fusion Network) - Porting to Pytorch

This repository is porting the [Keras Implementation](https://github.com/TUMFTM/CameraRadarFusionNet) to [PyTorch](https://pytorch.org/). 

This repository provides a neural network for object detection based on camera and radar data. It builds up on the work of [Keras RetinaNet](https://github.com/fizyr/keras-retinanet). 
The network performs a multi-level fusion of the radar and camera data within the neural network.
The network can be tested on the [nuScenes](https://www.nuscenes.org/) dataset, which provides camera and radar data along with 3D ground truth information.


## Requirements
- Linux Ubuntu (tested on version 18.04)
- CUDA 11.0
- Python 3.7
- PyTorch 1.7.1


# CRF-Net Usage
The network uses camera and radar inputs to detect objects. It can be used with the nuScenes dataset and extended to other radar and camera datasets. The nuScenes dataset can be downloaded [here](https://www.nuscenes.org/download).
Pretrained weights are provided [here](https://syncandshare.lrz.de/dl/fi9RrjqLXyLZFuhwjk9KiKjc/crf_net.h5 ) (270MB).
## Start Training
1. Create your desired [configuration](crfnet/configs/README.md) for the CRF-Net. Start by making a copy of the [default_config](crfnet/configs/default.cfg)
2. Execute `python train_crfnet.py`. This will train a model on a given dataset specified in the configs. The result will be stored in [saved_models](crfnet/saved_models) and the logs in [tb_logs](crfnet/tb_logs).
    * `--config <path to your config>` to use your config. Per default the config file found at [./configs/local.cfg](crfnet/configs/local.cfg) is used.
