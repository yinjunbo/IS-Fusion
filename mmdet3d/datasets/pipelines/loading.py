# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

import os.path as osp
from PIL import Image
import os
from typing import Any, Dict, Tuple

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesV2:  # v2: bevfusion
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if "img_filename" not in results:
            return results

        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))

        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        # Set initial values for default meta_keys
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadImageFromFileV2:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = Image.open(filename)
        # img_bytes = self.file_client.get(filename)
        # img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        # if self.to_float32:
        #     img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = [img]
        results['img_shape'] = img.size
        results['ori_shape'] = img.size
        results['img_fields'] = ['img']
        results["pad_shape"] = img.size
        results["scale_factor"] = 1.0

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class MyResize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        imgs = results['img']
        results['img'] = [imgs[i] for i in range(len(imgs))]
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results[key][idx],
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][idx].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results[key][idx],
                        results['scale'],
                        return_scale=True,
                        backend=self.backend)
                results[key][idx] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale, 1.0],
                                    dtype=np.float32)
            results['img_shape'] = img.shape
            # in case that there is no padding
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'][0].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

@PIPELINES.register_module()
class MyNormalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            for idx in range(len(results['img'])):
                results[key][idx] = mmcv.imnormalize(results[key][idx], self.mean, self.std,
                                                     self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class MyPad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                for idx in range(len(results[key])):
                    padded_img = mmcv.impad_to_multiple(
                        results[key][idx], self.size_divisor, pad_val=self.pad_val)
                    results[key][idx] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2])

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        # img = np.stack(
        #     [mmcv.imread(name, self.color_type) for name in filename], axis=-1)


        img = np.stack(
            [mmcv.imfrombytes(self.file_client.get(name), flag=self.color_type) \
                for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageFromFilesWaymo(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged', num_img=5):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.num_img = num_img

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_info']['filename']
        # img is of shape (h, w, c, num_views)
        img_list = []
        for name in filename:
            img = mmcv.imread(name, self.color_type)
            short_edge = img.shape[0]
            if short_edge != 1280:
                new_img = np.zeros((1280, 1920, 3))
                new_img[:short_edge] = img
                img = new_img
            img_list.append(img)
        img = np.stack(img_list, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False,
                 painting=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

        self.painting = painting
        if self.painting:
            self.predict_fun = paint2seg_func()

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _load_painting(self, points, painting_path):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            painting_bytes = self.file_client.get(painting_path)
            predict_idx = np.frombuffer(painting_bytes, dtype='uint8')
        except ConnectionError:
            mmcv.check_file_exist(painting_path)
            predict_idx = np.fromfile(painting_path, dtype='uint8')

        predict_idx_2d = self.predict_fun(predict_idx)
        sample_predict_label = np.eye(11, dtype='uint8')[predict_idx_2d]
        points = np.concatenate([points, sample_predict_label], axis=1)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts

                if self.painting:
                    painting_path = sweep['data_path'].split('/')
                    painting_path[-2] = 'LIDAR_TOP_MASK'
                    painting_path = '/'.join(i for i in painting_path)
                    if osp.isfile(painting_path):
                        points_sweep = self._load_painting(points_sweep, painting_path)
                    else:
                        continue

                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        if not self.painting:
            points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadForeground2D(object):
    """Load foreground info provided by 2D results

    The results will be added an item
        saved raw fg info = {

            'virtual_pixel_indices' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            'real_pixel_indices' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            note: shape above is (num_fg_pixels, 2), and range of indices is within original image scale 1600*900

            'virtual_points' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            'real_points' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            note: shape above is (num_fg_pixels, 3), virtual/real points are the corresponding points of foreground pixels in LiDAR system
        }
        results["foreground2D_info"] = {
            'fg_pixels' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            'fg_points' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
        }

    """
    def __init__(self, dataset='NuScenesDataset', **kwargs):
        self.dataset = dataset

    def _organize(self, fg_info):
        """
        Private function to select unique foreground pixels (and paired points)
        """
        if self.dataset == 'NuScenesDataset':
            cam_num = len(fg_info['virtual_pixel_indices'])
            fg_pixels, fg_points = [], []
            fg_real_pixels, fg_real_points = [], []
            for i in range(cam_num):
                # # random sample
                # indices = len(fg_info['virtual_pixel_indices'][i])
                # if indices > 2000:
                #     selected_indices = np.random.randint(indices, size=2000)
                # else:
                #     selected_indices = np.arange(indices)
                fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][:,:3], fg_info['real_pixel_indices'][i][:,:3]), axis=0)
                # fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][selected_indices,:2], fg_info['real_pixel_indices'][i][:,:2]), axis=0)
                # virtual_pixel_inds = fg_info['virtual_pixel_indices'][i]
                # virtual_pts_inds = fg_info['virtual_points'][i]
                # num_pts = virtual_pixel_inds.shape[0]
                # count = 1 if num_pts > 1 else num_pts

                # fg_pixel_indices = np.concatenate((virtual_pixel_inds[np.random.randint(num_pts, size=count), :], fg_info['real_pixel_indices'][i]), axis=0)

                if fg_info['virtual_points'][i].shape[1] == 3: # append label after xyz
                    fg_info['virtual_points'][i] = np.concatenate((fg_info['virtual_points'][i], fg_info['virtual_pixel_indices'][i][:,-11:]), axis=1)
                    fg_info['real_points'][i] = np.concatenate((fg_info['real_points'][i], fg_info['real_pixel_indices'][i][:,-11:]), axis=1)
                # fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
                fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
                # fg_points_set = np.concatenate((fg_info['virtual_points'][i][selected_indices], fg_info['real_points'][i]), axis=0)
                # fg_points_set = np.concatenate((virtual_pts_inds[np.random.randint(num_pts, size=count), :], fg_info['real_points'][i]), axis=0)

                # make timestamp for point cloud
                timestamp = np.zeros((fg_points_set.shape[0], 1))
                fg_points_set = np.concatenate((fg_points_set, timestamp), axis=1)

                fg_pixels.append(fg_pixel_indices)
                fg_points.append(fg_points_set)

                # also make timestamp for real point cloud
                fg_real_points_set = fg_info['real_points'][i]
                timestamp_real = np.zeros((fg_real_points_set.shape[0], 1))
                fg_real_points_set = np.concatenate((fg_real_points_set, timestamp_real), axis=1)

                fg_real_pixels.append(fg_info['real_pixel_indices'][i][:,:3])
                fg_real_points.append(fg_real_points_set)

            return dict(fg_pixels=fg_pixels, fg_points=fg_points, fg_real_pixels=fg_real_pixels, fg_real_points=fg_real_points)

        elif self.dataset == 'KittiDataset':
            if len(fg_info.keys()) == 4:
                fg_pixels = np.concatenate((fg_info['virtual_pixel_indices'], fg_info['real_pixel_indices']), axis=0)
                fg_points = np.concatenate((fg_info['virtual_points'], fg_info['real_points']), axis=0)
            else:
                fg_pixels = np.zeros((0,2))
                fg_points = np.zeros((0,6))

            return dict(fg_pixels=[fg_pixels], fg_points=[fg_points])

    def _make_point_class(self, fg_info):
        fg_points = fg_info['fg_points']
        cam_num = len(fg_points)
        point_class = get_points_type('LIDAR')
        for i in range(cam_num):
            fg_point = fg_points[i]
            fg_point = point_class(
                fg_point, points_dim=fg_point.shape[-1]
            )
            fg_points[i] = fg_point
        fg_info['fg_points'] = fg_points
        return fg_info

    def __call__(self, results):
        if self.dataset == 'NuScenesDataset':

            pts_filename = results['pts_filename']
            tokens = pts_filename.split('/')
            # might have bug when using absolute path. Just add ```fg_path='/'+fg_path```
            fg_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN_WITH_DEPTH", tokens[-1]+'.pkl.npy')
            # fg_path = os.path.join(*tokens[:-2], "FOREGROUND_DEPTH_COMPLETION", tokens[-1]+'.pkl.npy')
            fg_info = np.load(fg_path, allow_pickle=True).item()

            # # aggregate boundary info
            # fg_boundary_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_BD_200", tokens[-1]+'.pkl.npy')
            # fg_boundary_info = np.load(fg_boundary_path, allow_pickle=True).item()
            # for key in fg_info:
            #     if 'real' in key:
            #         continue
            #     fg_info[key] = list(map(lambda x,y: np.concatenate((x,y), axis=0), fg_info[key], fg_boundary_info[key]))

            fg_info = self._organize(fg_info)
            results["foreground2D_info"] = fg_info

            return results

        elif self.dataset == 'KittiDataset':

            # find the saved foreground points & pixels file
            pts_filename = results['pts_filename']
            tokens = pts_filename.split('/')
            fg_path = os.path.join(*tokens[:-2], "virtual_1NN", tokens[-1].split('.')[0]+'.npy')
            fg_info = np.load(fg_path, allow_pickle=True).item()
            fg_info = self._organize(fg_info)

            # make mmdet3d point class, as Kitti doesn't have multi-sweep settings
            fg_info = self._make_point_class(fg_info)
            results['foreground2D_info'] = fg_info

            return results

        else:
            raise NotImplementedError("foreground2D info of {} dataset is unavailable!".format(self.dataset))

@PIPELINES.register_module()
class LoadForeground2DFromMultiSweeps(object):
    """Load foreground info provided by 2D results from multiple sweeps

    """
    def __init__(self, dataset="NuScenesDataset", sweeps_num=10):
        self.dataset = dataset
        self.sweeps_num = sweeps_num

    def _organize(self, fg_info, results, sweep):
        """
        Private function to select unique foreground pixels (and paired points)
        """
        cam_num = len(fg_info['virtual_pixel_indices'])
        fg_pixels, fg_points = [], []
        fg_real_pixels, fg_real_points = [], []
        ts = results['timestamp']
        sweep_ts = sweep['timestamp'] / 1e6

        for i in range(cam_num):
            # random sampling
            # indices = len(fg_info['virtual_pixel_indices'][i])
            # if indices > 2000:
            #     selected_indices = np.random.randint(indices, size=2000)
            # else:
            #     selected_indices = np.arange(indices)

            fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][:,:3], fg_info['real_pixel_indices'][i][:,:3]), axis=0)
            # fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][selected_indices,:2], fg_info['real_pixel_indices'][i][:,:2]), axis=0)
            # virtual_pixel_inds = fg_info['virtual_pixel_indices'][i]
            # virtual_pts_inds = fg_info['virtual_points'][i]
            # num_pts = virtual_pixel_inds.shape[0]
            # count = 1 if num_pts > 1 else num_pts

            # fg_pixel_indices = np.concatenate((virtual_pixel_inds[np.random.randint(num_pts, size=count), :], fg_info['real_pixel_indices'][i]), axis=0)
            if fg_info['virtual_points'][i].shape[1] == 3: # append label after xyz
                fg_info['virtual_points'][i] = np.concatenate((fg_info['virtual_points'][i], fg_info['virtual_pixel_indices'][i][:,-11:]), axis=1)
                fg_info['real_points'][i] = np.concatenate((fg_info['real_points'][i], fg_info['real_pixel_indices'][i][:,-11:]), axis=1)
            # fg_points_set = np.concatenate((fg_info['virtual_points'][i][selected_indices], fg_info['real_points'][i]), axis=0)
            fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
            # fg_points_set = np.concatenate((virtual_pts_inds[np.random.randint(num_pts, size=count), :], fg_info['real_points'][i]), axis=0)

            # make timestamp for point cloud
            timestamp = np.zeros((fg_points_set.shape[0], 1))
            fg_points_set = np.concatenate((fg_points_set, timestamp), axis=1)
            fg_points_set[:, -1] = ts - sweep_ts

            fg_pixels.append(fg_pixel_indices)
            fg_points.append(fg_points_set)

            # also make timestamp for real point cloud
            fg_real_points_set = fg_info['real_points'][i]
            timestamp_real = np.zeros((fg_real_points_set.shape[0], 1))
            fg_real_points_set = np.concatenate((fg_real_points_set, timestamp_real), axis=1)
            fg_real_points_set[:, -1] = ts - sweep_ts / 1e-6

            fg_real_pixels.append(fg_info['real_pixel_indices'][i][:,:3])
            fg_real_points.append(fg_real_points_set)

        return dict(fg_pixels=fg_pixels, fg_points=fg_points, fg_real_pixels=fg_real_pixels, fg_real_points=fg_real_points)

    def _merge_sweeps(self, fg_info, sweep_fg_info, sweep):
        """
            fg_info and sweep_fg_info: dict like :
            {
                'fg_pixels' = [
                    np.array --> cam1, ..., np.array --> cam6
                ]
                'fg_points' = [
                    np.array --> cam1, ..., np.array --> cam6
                ]
            }
            sweep: dict of sweep info
        """
        fg_pixels, fg_points = fg_info['fg_pixels'], fg_info['fg_points']
        fg_real_pixels, fg_real_points = fg_info['fg_real_pixels'], fg_info['fg_real_points']
        sweep_fg_pixels, sweep_fg_points = sweep_fg_info['fg_pixels'], sweep_fg_info['fg_points']
        sweep_fg_real_pixels, sweep_fg_real_points = sweep_fg_info['fg_real_pixels'], sweep_fg_info['fg_real_points']

        if len(sweep_fg_points) == len(fg_points):
            cam_num = len(fg_pixels)

            for cam_id in range(cam_num):
                # merge fg_pixels and sweep_fg_pixels
                fg_pixel, sweep_fg_pixel = fg_pixels[cam_id], sweep_fg_pixels[cam_id]
                # might be a bug to be fixed in the future, i.e., misalignment between sweep pic and sample pic
                fg_pixel = np.concatenate((fg_pixel, sweep_fg_pixel), axis=0)
                fg_pixels[cam_id] = fg_pixel

                # merge fg_points and sweep_fg_points
                fg_point, sweep_fg_point = fg_points[cam_id], sweep_fg_points[cam_id]
                # Note: align sweep points with sample points

                sweep_fg_point[:,:3] = sweep_fg_point[:,:3] @ sweep['sensor2lidar_rotation'].T
                sweep_fg_point[:,:3] = sweep_fg_point[:,:3] + sweep['sensor2lidar_translation']
                fg_point = np.concatenate((fg_point, sweep_fg_point), axis=0)
                fg_points[cam_id] = fg_point

                fg_real_pixel, sweep_fg_real_pixel = fg_real_pixels[cam_id], sweep_fg_real_pixels[cam_id]

                fg_real_pixel = np.concatenate([fg_real_pixel, sweep_fg_real_pixel], axis=0)
                fg_real_pixels[cam_id] = fg_real_pixel

                fg_real_point, sweep_fg_real_point = fg_real_points[cam_id], sweep_fg_real_points[cam_id]
                sweep_fg_real_point[:,:3] = sweep_fg_real_point[:,:3] @ sweep['sensor2lidar_rotation'].T
                sweep_fg_real_point[:,:3] = sweep_fg_real_point[:,:3] + sweep['sensor2lidar_translation']
                fg_real_point = np.concatenate([fg_real_point, sweep_fg_real_point], axis=0)
                fg_real_points[cam_id] = fg_real_point

        else:
            print("##################################################")
            print(len(sweep_fg_points))
            print("##################################################")

        fg_info['fg_pixels'] = fg_pixels
        fg_info['fg_points'] = fg_points
        fg_info['fg_real_pixels'] = fg_real_pixels
        fg_info['fg_real_points'] = fg_real_points

        return fg_info

    def _make_point_class(self, fg_info):
        fg_points = fg_info['fg_points']
        cam_num = len(fg_points)
        point_class = get_points_type('LIDAR')
        for i in range(cam_num):
            fg_point = fg_points[i]
            fg_point = point_class(
                fg_point, points_dim=fg_point.shape[-1]
            )
            fg_points[i] = fg_point
        fg_info['fg_points'] = fg_points
        return fg_info


    def __call__(self, results):
        if self.dataset == "NuScenesDataset":
            fg_info = results["foreground2D_info"]

            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                pts_filename = sweep['data_path']
                tokens = pts_filename.split('/')
                sweep_fg_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN_WITH_DEPTH", tokens[-1]+'.pkl.npy')
                # sweep_fg_path = os.path.join(*tokens[:-2], "FOREGROUND_DEPTH_COMPLETION", tokens[-1]+'.pkl.npy')
                if os.path.exists(sweep_fg_path):
                    sweep_fg_info = np.load(sweep_fg_path, allow_pickle=True).item()
                    #     # merge boundary pts into
                    #     sweep_fg_boundary_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_BD_200", tokens[-1]+'.pkl.npy')
                    #     sweep_fg_boundary_info = np.load(sweep_fg_boundary_path, allow_pickle=True).item()
                    #     for key in sweep_fg_info:
                    #         if 'real' in key:
                    #             continue
                    #         sweep_fg_info[key] = list(map(lambda x,y: np.concatenate((x,y), axis=0), sweep_fg_info[key], sweep_fg_boundary_info[key]))

                    sweep_fg_info = self._organize(sweep_fg_info, results, sweep)
                    # merge sweep_fg_info with sample fg_info
                    fg_info = self._merge_sweeps(fg_info, sweep_fg_info, sweep)
                else:
                    continue

            # make mmdet3d LiDARPoint for each foreground 2D points
            fg_info = self._make_point_class(fg_info)

            results['foreground2D_info'] = fg_info

            return results


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str

def paint2seg_func():
    detection_mapping = {'car': 0, 'truck': 1, 'construction_vehicle': 2, 'bus': 3, 'trailer': 4, 'barrier': 5,
                         'motorcycle': 6, 'bicycle': 7, 'pedestrian': 8, 'traffic_cone': 9, 'others': 10}
    # self.class_3d_mapping = {'others': 0, 'barrier': 1, 'bicycle': 2, 'bus': 3, 'car': 4, 'construction_vehicle': 5,
    #                  'motorcycle': 6, 'pedestrian': 7, 'traffic_cone': 8, 'trailer': 9, 'truck': 10}
    class_2d_mapping = {'others': 0, 'car': 1, 'truck': 2, 'trailer': 3, 'bus': 4, 'construction_vehicle': 5,
                        'bicycle': 6, 'motorcycle': 7, 'pedestrian': 8, 'traffic_cone': 9, 'barrier': 10}
    paint2det = {class_2d_mapping[name]: detection_mapping[name] for name in class_2d_mapping}
    paint2seg_func = np.vectorize(paint2det.get)
    return paint2seg_func

@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 tanh_dim=None, # to normalize intensity and elongation in WaymoOpenDataset
                 shift_height=False,
                 use_color=False,
                 painting=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.tanh_dim = tanh_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        self.detection_mapping = {'car': 0, 'truck': 1, 'construction_vehicle': 2, 'bus': 3, 'trailer': 4, 'barrier': 5, 'motorcycle': 6, 'bicycle': 7, 'pedestrian': 8, 'traffic_cone': 9, 'others': 10}
        self.painting = painting
        if self.painting:
            self.predict_fun = paint2seg_func()

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def get_each_pc_gt_info(self, obj_points, class_name):
        assert class_name in list(self.detection_mapping.keys()),"class name not exist!"

        labels_2d = np.zeros((obj_points.shape[0], 11), np.float32)
        labels_2d[:, self.detection_mapping[class_name]] = 1
        # labels_3d = np.array([0, 1], dtype=np.float32).reshape(-1, 2).repeat(obj_points.shape[0], axis=0)
        return np.concatenate([obj_points, labels_2d], axis=1)

        # if self.painting == '2D':
        #     return np.concatenate([obj_points, labels_2d], axis=1)
        # elif self.painting == '3D':
        #     return np.concatenate([obj_points, labels_3d], axis=1)
        # else:
        #     return np.concatenate([obj_points,labels_2d,labels_3d], axis=1)

    def _load_painting(self, points, pts_filename, instance_name=None):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.painting=='gt_aug':
            points = self.get_each_pc_gt_info(points, instance_name)
        else:
            painting_path = pts_filename.split('/')
            painting_path[-2] = 'LIDAR_TOP_MASK'
            painting_path = '/'.join(i for i in painting_path)

            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)
            try:
                painting_bytes = self.file_client.get(painting_path)
                predict_idx = np.frombuffer(painting_bytes, dtype='uint8')
            except ConnectionError:
                mmcv.check_file_exist(painting_path)
                predict_idx = np.fromfile(painting_path, dtype='uint8')

            predict_idx_2d = self.predict_fun(predict_idx)
            sample_predict_label = np.eye(11, dtype='uint8')[predict_idx_2d]
            points = np.concatenate([points, sample_predict_label], axis=1)
        return points

    def __call__(self, results, instance_name=None):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.tanh_dim is not None:
            assert isinstance(self.tanh_dim, list)
            assert max(self.tanh_dim) < points.shape[1]
            assert min(self.tanh_dim) > 2
            points[:, self.tanh_dim] = np.tanh(points[:, self.tanh_dim])

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        if self.painting:
            points = self._load_painting(points, pts_filename, instance_name)

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str
