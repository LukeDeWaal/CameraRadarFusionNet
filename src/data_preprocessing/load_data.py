import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from nuscenes_devkit.nuscenes import NuScenes
from nuscenes_devkit.nuimages import NuImages
from src.defines import ROOT_DIR, DATA_DIR, MODEL_DIR


class NuscenesDataset(Dataset):
    """
    Nuscenes Image and RADAR Datasets
    """
    def __init__(self, data_root: str, **kwargs):
        self.NS = NuScenes(dataroot=data_root, **kwargs)
        self.NI = NuImages(dataroot=data_root, **kwargs)



if __name__ == "__main__":
    NS = NuScenes(version='v1.0-mini', dataroot=os.path.join(DATA_DIR, 'mini'))
    NI = NuImages(version='v1.0-mini', dataroot=os.path.join(DATA_DIR, 'mini'))

    scene = NS.scene[0]
    token = scene['first_sample_token']
    sample = NS.get('sample', token)
    sensor = 'CAM_FRONT'
    cam_front_data = NS.get('sample_data', sample['data'][sensor])
    NS.render_sample_data(cam_front_data['token'])
