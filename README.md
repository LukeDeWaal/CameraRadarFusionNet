# CRF-Net for Object Detection (Camera and Radar Fusion Network) - Porting to Pytorch

This repository is porting the [Keras Implementation](https://github.com/TUMFTM/CameraRadarFusionNet) to [PyTorch](https://pytorch.org/). 

This repository provides a neural network for object detection based on camera and radar data. It builds up on the work of [Keras RetinaNet](https://github.com/fizyr/keras-retinanet). 
The network performs a multi-level fusion of the radar and camera data within the neural network.
The network can be tested on the [nuScenes](https://www.nuscenes.org/) dataset, which provides camera and radar data along with 3D ground truth information.


## Requirements
- Linux Ubuntu (tested on version 18.04)
- CUDA 11.0
- Anaconda Python 3.7
- PyTorch 1.7.1
- [Nuscenes Devkit for Conda](https://github.com/LukeDeWaal/nuscenes_devkit) 


# CRF-Net Usage
The network uses camera and radar inputs to detect objects. It can be used with the nuScenes dataset and extended to other radar and camera datasets. The nuScenes dataset can be downloaded [here](https://www.nuscenes.org/download).
Pretrained weights are provided [here](https://syncandshare.lrz.de/dl/fi9RrjqLXyLZFuhwjk9KiKjc/crf_net.h5 ) (270MB).

