import numpy as np
import torch
from torch.utils.data import DataLoader

from data_preprocessing.tools.anchor import guess_shapes
from data_preprocessing.tools.anchor_calc import anchor_targets_bbox
from data_preprocessing.tools.data_pipeline import preprocess_image
from data_preprocessing.tools.transform import random_transform_generator
from defines import *
from load_data import NuscenesDataset


if __name__ == "__main__":
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.0,
    )

    common_args = {
        'batch_size': 1,
        'config': None,
        'image_min_side': cfg.image_size[0],
        'image_max_side': cfg.image_size[1],
        'filter_annotations_enabled': False,
        'preprocess_image': preprocess_image,
        'normalize_radar': cfg.normalize_radar,
        'camera_dropout': cfg.dropout_image,
        'radar_dropout': cfg.dropout_radar,
        'channels': cfg.channels,
        'distance': cfg.distance_detection,
        'sample_selection': cfg.sample_selection,
        'only_radar_annotated': cfg.only_radar_annotated,
        'n_sweeps': cfg.n_sweeps,
        'noise_filter': cfg.noise_filter_cfg,
        'noise_filter_threshold': cfg.noise_filter_threshold,
        'noisy_image_method': cfg.noisy_image_method,
        'noise_factor': cfg.noise_factor,
        'perfect_noise_filter': cfg.noise_filter_perfect,
        'radar_projection_height': cfg.radar_projection_height,
        'noise_category_selection': None if cfg.class_weights is None else cfg.class_weights.keys(),
        'inference': cfg.inference,
        'anchor_params': None,
    }

    class_to_color = {
        'bg': np.array([0, 0, 0]) / 255,
        'human.pedestrian.adult': np.array([34, 114, 227]) / 255,
        'vehicle.bicycle': np.array([0, 182, 0]) / 255,
        'vehicle.bus': np.array([84, 1, 71]) / 255,
        'vehicle.car': np.array([189, 101, 0]) / 255,
        'vehicle.motorcycle': np.array([159, 157, 156]) / 255,
        'vehicle.trailer': np.array([0, 173, 162]) / 255,
        'vehicle.truck': np.array([89, 51, 0]) / 255,
    }

    data_generator = NuscenesDataset(r"v1.0-trainval", EXT_FULL_DIR, 'CAM_FRONT', radar_input_name='RADAR_FRONT',
                                     scene_indices=None,
                                     category_mapping=cfg.category_mapping,
                                     transform_generator=None,
                                     shuffle_groups=False,
                                     compute_anchor_targets=anchor_targets_bbox,
                                     compute_shapes=guess_shapes,
                                     verbose=False,
                                     threading=True,
                                     timer=True,
                                     **common_args)

    CORES = 6
    dataloader = DataLoader(
        data_generator,
        batch_size=cfg.batchsize,
        shuffle=False,
        num_workers=CORES,
        prefetch_factor=1
    )

    for i_batch, sample_batched in enumerate(dataloader):
        input, target = sample_batched
        print(f"{str(i_batch) + ' '*(1-len(str(i_batch)))}   "
              f"        {str(tuple(input.shape)) + ' '*(1-len(str(tuple(input.shape))))}    {[tuple(t.shape) for t in target]}")