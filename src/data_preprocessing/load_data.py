import os, sys, math
import warnings

import numpy as np
import cv2
from typing import Union
import random
import time as tm
from nuscenes.utils.data_classes import Box
from torch.utils.data import Dataset
from data_preprocessing.tools.visualization import draw_boxes

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(*file_dir.split(os.path.sep)[:-1]))

import pyximport
pyximport.install()

from nuscenes_devkit.nuscenes import NuScenes, RadarPointCloud

from data_preprocessing.tools.anchor import guess_shapes, anchors_for_shape, compute_gt_annotations
from data_preprocessing.tools.anchor_calc import anchor_targets_bbox
from data_preprocessing.tools.nuscenes_helper import calc_mask, get_sensor_sample_data
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility, points_in_box
from data_preprocessing.tools.utils import imageplus_creation, create_spatial_point_array, noisy, create_imagep_visualization
from defines import *
from data_preprocessing.tools.data_pipeline import preprocess_image, TransformParameters, preprocess_image_inverted, adjust_transform_for_image, \
    apply_transform, resize_image
from data_preprocessing.tools.transform import random_transform_generator, transform_aabb
from data_preprocessing.tools import radar


class NuscenesDataset(Dataset):
    """
    Nuscenes Frontal Image and RADAR Dataset
    """

    def __init__(self,
                 version: str,
                 data_root: str,
                 cam_sensor: str,
                 #radar_sensor: str,
                 scene_indices=None,
                 channels=[0, 1, 2],
                 category_mapping=None,
                 radar_input_name=None,
                 radar_width=None,
                 image_radar_fusion=True,
                 camera_dropout=0.0,
                 radar_dropout=0.0,
                 normalize_radar=False,
                 sample_selection=False,
                 only_radar_annotated=False,
                 n_sweeps=1,
                 noise_filter=None,
                 noise_filter_threshold=0.5,
                 noisy_image_method=None,
                 noise_factor=0,
                 perfect_noise_filter=False,
                 noise_category_selection=None,
                 inference=False,
                 transform_generator=None,
                 batch_size=1,
                 group_method=None,  # one of 'none', 'random', 'ratio'
                 shuffle_groups=False,
                 image_min_side=800,
                 image_max_side=1333,
                 transform_parameters=None,
                 compute_anchor_targets=None,
                 compute_shapes=None,
                 preprocess_image=None,
                 filter_annotations_enabled=True,
                 config=None,
                 distance=False,
                 radar_projection_height=3,
                 anchor_params=None,
                 verbose=False,
                 timer=False,
                 *args,
                 **kwargs
                 ):

        print("Dataset Initialization :", end='')
        t1 = tm.time()
        self.version = version
        self.data_root = data_root
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False, **kwargs)
        self.cam_sensor = cam_sensor
        self.radar_sensor = radar_input_name

        self.dropout_chance = 0.0
        self.radar_sensors = ['RADAR_FRONT']
        self.camera_sensors = ['CAM_FRONT']
        self.labels = {}
        self.image_data = dict()
        self.classes, self.labels = self._get_class_label_mapping([c['name'] for c in self.nusc.category], category_mapping)
        self.channels = channels
        self.radar_channels = [ch for ch in channels if ch >= 3]
        self.image_channels = [ch for ch in channels if ch < 3]
        self.normalize_bbox = False  # True for normalizing the bbox to [0,1]
        self.radar_input_name = radar_input_name
        self.radar_width = radar_width
        self.radar_dropout = radar_dropout
        self.camera_dropout = camera_dropout
        self.sample_selection = sample_selection
        self.only_radar_annotated = only_radar_annotated
        self.n_sweeps = n_sweeps
        self.noisy_image_method = noisy_image_method
        self.noise_factor = noise_factor
        self.cartesian_uncertainty = (0, 0, 0)  # meters
        self.angular_uncertainty = 0.0  # degree
        self.inference = inference
        self.timer = timer

        self.image_min_side = image_min_side
        self.image_max_side = image_max_side

        # assign functions
        self.image_radar_fusion = image_radar_fusion
        self.normalize_radar = normalize_radar

        # Optional imports
        self.radar_array_creation = None
        if self._is_image_plus_enabled() or self.camera_dropout > 0.0:
            # Installing vizdom is required
            self.image_plus_creation = imageplus_creation
            self.radar_array_creation = create_spatial_point_array

        self.noise_filter_threshold = noise_filter_threshold
        self.perfect_noise_filter = perfect_noise_filter
        self.noise_category_selection = noise_category_selection

        if noise_filter:
            raise NotImplementedError('Neural Filter not in opensource repository ')
        else:
            self.noise_filter = None

        # Create all sample tokens
        self.sample_tokens = {}
        prog = 0
        #progbar = progressbar.ProgressBar(prefix='Initializing data generator: ')
        skip_count = 0

        # Resolve sample indexing
        if scene_indices is None:
            # We are using all scenes
            scene_indices = range(len(self.nusc.scene))

        assert hasattr(scene_indices, '__iter__'), "Iterable object containing sample indices expected"

        for scene_index in scene_indices:
            first_sample_token = self.nusc.scene[scene_index]['first_sample_token']
            nbr_samples = self.nusc.scene[scene_index]['nbr_samples']

            curr_sample = self.nusc.get('sample', first_sample_token)

            for _ in range(nbr_samples):
                self.sample_tokens[prog] = curr_sample['token']
                if curr_sample['next']:
                    next_token = curr_sample['next']
                    curr_sample = self.nusc.get('sample', next_token)
                prog += 1
                #progbar.update(prog)

        if self.sample_selection: print("\nSkipped {} samples due to zero annotations".format(skip_count))
        # Create all annotations and put into image_data
        self.image_data = {image_index: None for image_index in self.sample_tokens}

        self.transform_generator = transform_generator
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.transform_parameters = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes = compute_shapes
        self.preprocess_image = preprocess_image
        self.filter_annotations_enabled = filter_annotations_enabled
        self.config = config
        self.distance = distance
        self.radar_projection_height = radar_projection_height
        self.anchor_params = anchor_params

        self.verbose = verbose

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

        t2 = tm.time()
        dt = t2 - t1
        print(f" {dt} s")


    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx: Union[int, np.int32, np.int64, np.uint32, np.uint64]):
        if isinstance(idx, (int, np.int32, np.int64, np.uint32, np.uint64)):
            pass
        else:
            raise TypeError(f"Expected an integer type, got {type(idx)}")

        t1 = tm.time()
        group = self.groups[idx]
        inputs, targets = self.compute_input_output(group, inference=self.inference)
        if self.batch_size == 1:
            inputs = inputs[0, ...]
            targets = [t[0, ...] for t in targets]
        t2 = tm.time()
        dt = t2-t1
        #if self.verbose: print(f"Data Item Retrieval: {t2-t1} s")
        if self.timer:
            print(round(dt, 5), "[s]", end='')

        return inputs, targets

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    @staticmethod
    def _get_class_label_mapping(category_names, category_mapping):
        """
        :param category_mapping: [dict] Map from original name to target name. Subsets of names are supported.
            e.g. {'pedestrian' : 'pedestrian'} will map all pedestrian types to the same label

        :returns:
            [0]: [dict of (str, int)] mapping from category name to the corresponding index-number
            [1]: [dict of (int, str)] mapping from index number to category name
        """
        # Initialize local variables
        original_name_to_label = {}
        original_category_names = category_names.copy()
        original_category_names.append('bg')
        if category_mapping is None:
            # Create identity mapping and ignore no class
            category_mapping = dict()
            for cat_name in category_names:
                category_mapping[cat_name] = cat_name

        # List of unique class_names
        selected_category_names = set(category_mapping.values())  # unordered
        selected_category_names = list(selected_category_names)
        selected_category_names.sort()  # ordered

        # Create the label to class_name mapping
        label_to_name = {label: name for label, name in enumerate(selected_category_names)}
        label_to_name[len(label_to_name)] = 'bg'  # Add the background class

        # Create original class name to label mapping
        for label, label_name in label_to_name.items():

            # Looking for all the original names that are adressed by label name
            targets = [original_name for original_name in original_category_names if label_name in original_name]

            # Assigning the same label for all adressed targets
            for target in targets:
                # Check for ambiguity
                assert target not in original_name_to_label.keys(), 'ambigous mapping found for (%s->%s)' % (
                target, label_name)

                # Assign label to original name
                # Some label_names will have the same label, which is totally fine
                original_name_to_label[target] = label

        # Check for correctness
        actual_labels = original_name_to_label.values()
        expected_labels = range(0, max(actual_labels) + 1)  # we want to start labels at 0
        assert all([label in actual_labels for label in expected_labels]), 'Expected labels do not match actual labels'

        return original_name_to_label, label_to_name

    def _is_image_plus_enabled(self):
        """
        True if image radar fusion is enabled and
        radar channels are requested.
        """
        r = 0 in self.channels
        g = 1 in self.channels
        b = 2 in self.channels
        return self.image_radar_fusion and len(self.channels) > r + g + b

    def size(self):
        """ Size of the dataset.
        """
        return len(self.sample_tokens)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.labels)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def inv_label_to_name(self, name):
        """ Map name to label.
        """
        class_dict = {y: x for x, y in self.labels.items()}
        return class_dict[name]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        All images of nuscenes dataset have the same aspect ratio which is 16/9
        """
        # All images of nuscenes dataset have the same aspect ratio
        return 16 / 9
        # sample_token = self.sample_tokens[image_index]
        # sample = self.nusc.get('sample', sample_token)

        # image_sample = self.load_sample_data(sample, camera_name)
        # return float(image_sample.shape[1]) / float(image_sample.shape[0])

    def load_radar_array(self, sample_index, target_width):
        # Initialize local variables
        if not self.radar_array_creation:
            self.radar_array_creation = create_spatial_point_array

        radar_name = self.radar_sensors[0]
        camera_name = self.camera_sensors[0]

        # Gettign data from nuscenes database
        sample_token = self.sample_tokens[sample_index]
        sample = self.nusc.get('sample', sample_token)

        # Grab the front camera and the radar sensor.
        radar_token = sample['data'][radar_name]
        camera_token = sample['data'][camera_name]
        image_target_shape = (self.image_min_side, self.image_max_side)

        # Create the array
        radar_sample = self.load_sample_data(sample, radar_name)  # Load samples from disk
        radar_array = self.radar_array_creation(self.nusc, radar_sample, radar_token, camera_token,
                                                target_width=target_width)

        return radar_array

    def set_noise_factor(self, noise_factor):
        """
        This function turns off the noise factor: It is useful for rendering.
        """
        self.noise_factor = noise_factor

    def load_image(self, image_index):
        """
        Returns the image plus from given image and radar samples.
        It takes the requested channels into account.

        :param sample_token: [str] the token pointing to a certain sample

        :returns: imageplus
        """
        # Initialize local variables
        radar_name = self.radar_sensors[0]
        camera_name = self.camera_sensors[0]

        # Gettign data from nuscenes database
        sample_token = self.sample_tokens[image_index]
        sample = self.nusc.get('sample', sample_token)

        # Grab the front camera and the radar sensor.
        radar_token = sample['data'][radar_name]
        camera_token = sample['data'][camera_name]
        image_target_shape = (self.image_min_side, self.image_max_side)

        # Load the image
        image_sample = self.load_sample_data(sample, camera_name)

        # Add noise to the image if enabled
        if self.noisy_image_method is not None and self.noise_factor > 0:
            image_sample = noisy(self.noisy_image_method, image_sample, self.noise_factor)

        if self._is_image_plus_enabled() or self.camera_dropout > 0.0:

            # Parameters
            kwargs = {
                'pointsensor_token': radar_token,
                'camera_token': camera_token,
                'height': (0, self.radar_projection_height),
                'image_target_shape': image_target_shape,
                'clear_radar': np.random.rand() < self.radar_dropout,
                'clear_image': np.random.rand() < self.camera_dropout,
            }

            # Create image plus
            # radar_sample = self.load_sample_data(sample, radar_name) # Load samples from disk

            # Get filepath
            if self.noise_filter:
                required_sweep_count = self.n_sweeps + self.noise_filter.num_sweeps_required - 1
            else:
                required_sweep_count = self.n_sweeps

            # sd_rec = self.nusc.get('sample_data', sample['data'][sensor_channel])
            sensor_channel = radar_name
            pcs, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, sensor_channel, \
                                                              sensor_channel, nsweeps=required_sweep_count,
                                                              min_distance=0.0)
            if isinstance(pcs, list):
                pass
            else:
                pcs = [pcs]

            if self.noise_filter:
                # fill up with zero sweeps
                for _ in range(required_sweep_count - len(pcs)):
                    pcs.insert(0, RadarPointCloud(np.zeros(shape=(RadarPointCloud.nbr_dims(), 0))))

            radar_sample = [radar.enrich_radar_data(pc.points) for pc in pcs]

            if self.noise_filter:
                ##### Filter the pcs #####
                radar_sample = list(self.noise_filter.denoise(radar_sample, self.n_sweeps))

            if len(radar_sample) == 0:
                radar_sample = np.zeros(shape=(len(radar.channel_map), 0))
            else:
                ##### merge pcs into single radar samples array #####
                radar_sample = np.concatenate(radar_sample, axis=-1)

            radar_sample = radar_sample.astype(dtype=np.float32)

            if self.perfect_noise_filter:
                cartesian_uncertainty = 0.5  # meters
                angular_uncertainty = math.radians(1.7)  # degree
                category_selection = self.noise_category_selection

                nusc_sample_data = self.nusc.get('sample_data', radar_token)
                radar_gt_mask = calc_mask(nusc=self.nusc, nusc_sample_data=nusc_sample_data,
                                          points3d=radar_sample[0:3, :], \
                                          tolerance=cartesian_uncertainty, angle_tolerance=angular_uncertainty, \
                                          category_selection=category_selection)

                # radar_sample = radar_sample[:, radar_gt_mask.astype(np.bool)]
                radar_sample = np.compress(radar_gt_mask, radar_sample, axis=-1)

            if self.normalize_radar:
                # we need to noramlize
                # : use preprocess method analog to image preprocessing
                sigma_factor = int(self.normalize_radar)
                for ch in range(3, radar_sample.shape[0]):  # neural fusion requires x y and z to be not normalized
                    norm_interval = (-127.5, 127.5)  # caffee mode is default and has these norm interval for img
                    radar_sample[ch, :] = radar.normalize(ch, radar_sample[ch, :], normalization_interval=norm_interval,
                                                          sigma_factor=sigma_factor)

            img_p_full = self.image_plus_creation(self.nusc, image_data=image_sample, radar_data=radar_sample, **kwargs)

            # reduce to requested channels
            # self.channels = [ch - 1 for ch in self.channels] # Shift channels by 1, cause we have a weird convetion starting at 1
            input_data = img_p_full[:, :, self.channels]

        else:  # We are not in image_plus mode
            # Only resize, because in the other case this is contained in image_plus_creation
            input_data = cv2.resize(image_sample, image_target_shape[::-1])

        return input_data

    def load_sample_data(self, sample, sensor_channel):
        """
        This function takes the token of a sample and a sensor sensor_channel and returns the according data

        Radar format: <np.array>
            - Shape: 18 x n
            - Semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0

        Image format: <np.array>
            - Shape: h x w x 3
            - Values: [0,255]
            - Channels: RGB
        """
        return get_sensor_sample_data(self.nusc, sample, sensor_channel, dtype=np.float32, size=None)

    def create_annotations(self, sample_token, sensor_channels):
        """
        Create annotations for the the given sample token.

        1 bounding box vector contains:


        :param sample_token: the sample_token to get the annotation for
        :param sensor_channels: list of channels for cropping the labels, e.g. ['CAM_FRONT', 'RADAR_FRONT']
            This works only for CAMERA atm

        :returns:
            annotations dictionary:
            {
                'labels': [] # <list of n int>
                'bboxes': [] # <list of n x 4 float> [xmin, ymin, xmax, ymax]
                'distances': [] # <list of n float>  Center of box given as x, y, z.
                'visibilities': [] # <list of n float>  Visibility of annotated object
            }
        """

        if any([s for s in sensor_channels if 'RADAR' in s]):
            print("[WARNING] Cropping to RADAR is not supported atm")
            sensor_channels = [c for c in sensor_channels if 'CAM' in sensor_channels]

        sample = self.nusc.get('sample', sample_token)
        annotations_count = 0
        annotations = {
            'labels': [],  # <list of n int>
            'bboxes': [],  # <list of n x 4 float> [xmin, ymin, xmax, ymax]
            'distances': [],  # <list of n float>  Center of box given as x, y, z.
            'visibilities': [],
            'num_radar_pts': []  # <list of n int>  number of radar points that cover that annotation
        }

        # Camera parameters
        for selected_sensor_channel in sensor_channels:
            sd_rec = self.nusc.get('sample_data', sample['data'][selected_sensor_channel])

            # Create Boxes:
            _, boxes, camera_intrinsic = self.nusc.get_sample_data(sd_rec['token'], box_vis_level=BoxVisibility.ANY)
            imsize_src = (sd_rec['width'], sd_rec['height'])  # nuscenes has (width, height) convention

            bbox_resize = [1. / sd_rec['height'], 1. / sd_rec['width']]
            if not self.normalize_bbox:
                bbox_resize[0] *= float(self.image_min_side)
                bbox_resize[1] *= float(self.image_max_side)

            # Create labels for all boxes that are visible
            for box in boxes:

                # Add labels to boxes
                if box.name in self.classes:
                    box.label = self.classes[box.name]
                    # Check if box is visible and transform box to 1D vector
                    if box_in_image(box=box, intrinsic=camera_intrinsic, imsize=imsize_src,
                                    vis_level=BoxVisibility.ANY):

                        ## Points in box method for annotation filterS
                        # check if bounding box has an according radar point
                        if self.only_radar_annotated == 2:

                            pcs, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, self.radar_sensors[0], \
                                                                              selected_sensor_channel,
                                                                              nsweeps=self.n_sweeps, min_distance=0.0,
                                                                              merge=False)

                            for pc in pcs:
                                pc.points = radar.enrich_radar_data(pc.points)

                            if len(pcs) > 0:
                                radar_sample = np.concatenate([pc.points for pc in pcs], axis=-1)
                            else:
                                print("[WARNING] only_radar_annotated=2 and sweeps=0 removes all annotations")
                                radar_sample = np.zeros(shape=(len(radar.channel_map), 0))
                            radar_sample = radar_sample.astype(dtype=np.float32)

                            mask = points_in_box(box, radar_sample[0:3, :])
                            if True not in mask:
                                continue

                        def box2d_calc(box: Box):
                            corners = box.corners()
                            outer_corners = [np.min(corners[0,:]),np.min(corners[1,:]), np.max(corners[0,:]), np.max(corners[1,:])]
                            return outer_corners

                        # If visible, we create the corresponding label
                        #box2d = box.box2d(camera_intrinsic)  # returns [xmin, ymin, xmax, ymax]
                        box2d = box2d_calc(box)
                        box2d[0] *= bbox_resize[1]
                        box2d[1] *= bbox_resize[0]
                        box2d[2] *= bbox_resize[1]
                        box2d[3] *= bbox_resize[0]

                        annotations['bboxes'].insert(annotations_count, box2d)
                        annotations['labels'].insert(annotations_count, box.label)
                        annotations['num_radar_pts'].insert(annotations_count,
                                                            self.nusc.get('sample_annotation', box.token)[
                                                                'num_radar_pts'])

                        distance = (box.center[0] ** 2 + box.center[1] ** 2 + box.center[2] ** 2) ** 0.5
                        annotations['distances'].insert(annotations_count, distance)
                        annotations['visibilities'].insert(annotations_count, int(
                            self.nusc.get('sample_annotation', box.token)['visibility_token']))
                        annotations_count += 1
                else:
                    # The current name has been ignored
                    # print(f"[SKIP] {sample_token}")
                    pass

        annotations['labels'] = np.array(annotations['labels'])
        annotations['bboxes'] = np.array(annotations['bboxes'])
        annotations['distances'] = np.array(annotations['distances'])
        annotations['num_radar_pts'] = np.array(annotations['num_radar_pts'])
        annotations['visibilities'] = np.array(annotations['visibilities'])

        # num_radar_pts mathod for annotation filter
        if self.only_radar_annotated == 1:

            anns_to_keep = np.where(annotations['num_radar_pts'])[0]

            for key in annotations:
                annotations[key] = annotations[key][anns_to_keep]

        return annotations

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        annotations = self.image_data[image_index]

        if annotations is None:
            sample_token = self.sample_tokens[image_index]
            annotations = self.create_annotations(sample_token, self.camera_sensors)

            self.image_data[image_index] = annotations

        return annotations

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        # for annotations in annotations_group:
        #     assert(isinstance(annotations, dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
        #     assert('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
        #     assert('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def compute_input_output(self, group, inference=False):
        """
        Extends the basic function with the capability to
        add radar input data to the input batch.
        """

        image_group = self.load_image_group(group)

        if inference:
            annotations_group = None
        else:
            annotations_group = self.load_annotations_group(group)

            # Load annotation
            if self.filter_annotations_enabled:
                # check validity of annotations
                image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

            # randomly transform data
            #image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        if annotations_group:
            # compute network targets
            targets = self.compute_targets(image_group, annotations_group)
        else:
            targets = None


        # if self.radar_input_name:
        #     # Load radar data
        #     radar_input_batch = []
        #     for sample_index in group:
        #         radar_array = self.load_radar_array(sample_index, target_width=self.radar_width)
        #         radar_input_batch.append(radar_array)
        #
        #     radar_input_batch = np.array(radar_input_batch)
        #
        #     tmp_inputs = (
        #         inputs,
        #         radar_input_batch
        #     )

        return inputs, targets

    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image,
                                                       self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            # annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index],
                                                                                             annotations_group[index])

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        if self.preprocess_image:
            image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        if annotations:
            # apply resizing to annotations too
            annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = image.astype(np.float32)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        if annotations_group:
            assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            anns_group_entry = annotations_group[index] if annotations_group else None
            image_group[index], preprocessed_anns = self.preprocess_group_entry(image_group[index], anns_group_entry)

            if annotations_group:
                annotations_group[index] = preprocessed_anns

        return image_group, annotations_group

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        # image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())
        image_batch = np.zeros((len(image_group),) + max_shape, dtype=np.float32)

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        # if keras.backend.image_data_format() == 'channels_first':
        #     image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = self.anchor_params
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes(),
            distance=self.distance
        )

        return list(batches)


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
        'batch_size': cfg.batchsize,
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
        'bg': np.array([0, 0, 0])/255,
        'human.pedestrian.adult': np.array([34, 114, 227]) / 255,
        'vehicle.bicycle': np.array([0, 182, 0])/255,
        'vehicle.bus': np.array([84, 1, 71])/255,
        'vehicle.car': np.array([189, 101, 0]) / 255,
        'vehicle.motorcycle': np.array([159, 157,156])/255,
        'vehicle.trailer': np.array([0, 173, 162])/255,
        'vehicle.truck': np.array([89, 51, 0])/255,
        }

    data_generator = NuscenesDataset(r"v1.0-trainval", EXT_FULL_DIR, 'CAM_FRONT',
                                     radar_input_name='RADAR_FRONT',
                                     scene_indices=None,
                                     category_mapping=cfg.category_mapping,
                                     transform_generator=transform_generator,
                                     shuffle_groups=False,
                                     compute_anchor_targets=anchor_targets_bbox,
                                     compute_shapes=guess_shapes,
                                     verbose=True,
                                     threading=False,
                                     **common_args)


    bboxes = True
    debug = True
    visual = False

    shitty_data = []
    i = 10
    while i < 60:#len(data_generator):
        print("Sample: ", i)
        # Get the data
        inputs, targets = data_generator[i]
        img = inputs[0] if isinstance(inputs, tuple) else inputs

        assert img.shape[0] == data_generator.batch_size
        img = img[0, ...]
        img = preprocess_image_inverted(img)
        ann = data_generator.load_annotations_group(data_generator.groups[i])

        assert img.shape[0] == common_args['image_min_side']
        assert img.shape[1] == common_args['image_max_side']
        # assert img.shape[2] == len(common_args['channels'])

        # Turn data into vizualizable format
        viz = create_imagep_visualization(img, draw_circles=False, cfg=cfg, radar_lines_opacity=0.9)


        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        lineType = 1

        if debug:
            ## Positive Anchor Visualization

            if isinstance(ann, list):
                for j, a in enumerate(ann):
                    if bool(sum([np.sum(v) for v in a.values()])):
                        anchors = anchors_for_shape(viz.shape, anchor_params=common_args['anchor_params'])
                        positive_indices, _, max_indices = compute_gt_annotations(anchors, a['bboxes'])
                        draw_boxes(viz, anchors[positive_indices], (255, 255, 0), thickness=1)

                        ## Data Augmentation
                        viz, ann[j] = data_generator.random_transform_group_entry(viz, a)

            elif isinstance(ann, dict):
                if bool(sum([np.sum(v) for v in ann.values()])):
                    anchors = anchors_for_shape(viz.shape, anchor_params=common_args['anchor_params'])
                    positive_indices, _, max_indices = compute_gt_annotations(anchors, ann['bboxes'])
                    draw_boxes(viz, anchors[positive_indices], (255, 255, 0), thickness=1)

                    ## Data Augmentation
                    viz, ann = data_generator.random_transform_group_entry(viz, ann)


        if bboxes:
            if isinstance(ann, list):
                outer_lim = len(ann)
                anns = ann
            else:
                anns = [ann]
                outer_lim = 1

            for k in range(outer_lim):
                for a in range(len(anns[k]['bboxes'])):
                    label_name = data_generator.label_to_name(anns[k]['labels'][a])
                    dist = anns[k]['distances'][a]

                    if label_name in class_to_color:
                        color = class_to_color[label_name] * 255
                    else:
                        color = class_to_color['bg']

                    p1 = (int(anns[k]['bboxes'][a][0]), int(anns[k]['bboxes'][a][1]))  # Top left
                    p2 = (int(anns[k]['bboxes'][a][2]), int(anns[k]['bboxes'][a][3]))  # Bottom right
                    cv2.rectangle(viz, p1, p2, color, 1)

                    textLabel = '{0}: {1:3.1f} {2}'.format(label_name.split('.', 1)[-1], dist, 'm')

                    (retval, baseLine) = cv2.getTextSize(textLabel, font, fontScale, 1)

                    textOrg = p1

                    cv2.rectangle(viz, (textOrg[0] - 1, textOrg[1] + baseLine - 1),
                                  (textOrg[0] + retval[0] + 1, textOrg[1] - retval[1] - 1), color, -1)
                    cv2.putText(viz, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1)

        if visual:
            # Visualize data
            cv2.imshow("Nuscenes Data Visualization", viz)
            # cv2.imwrite('./ground_truth_selected/' + str(i).zfill(4) +'.png', viz*255)
            key = cv2.waitKey(0)
            if key == ord('p'):  # previous image
                i = i - 1
            elif key == ord('s'):
                print("saving image")
                cv2.imwrite(f"{os.path.expanduser('~')}/Pictures/tmp/saved_img_{i}.png", viz)
            elif key == ord('n'):
                print("%c -> jump to next scene" % key)
                i = i + 40
            elif key == ord('m'):
                print("%c -> jump to previous scene" % key)
                i = i - 40
            elif key == ord('q'):
                break
            else:
                i = i + 1
        else:
            i += 1

        i = max(i, 0)
