import copy
import cv2
import mmcv
import numpy as np
import os
from PIL import Image

from mmdet3d.core.bbox import box_np_ops, LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet.datasets import PIPELINES
from ..builder import OBJECTSAMPLERS

class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 painting=None,
                 classes=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3])):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}

        self.painting = painting
        if painting:
            points_loader.update(painting='gt_aug')

        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_bboxes, gt_labels, img=None, painting=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)

                if self.painting:
                    s_points = self.points_loader(results, info['name'])['points']
                else:
                    s_points = self.points_loader(results)['points']

                s_points.translate(info['box3d_lidar'][:3])

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)
            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples


@OBJECTSAMPLERS.register_module()
class MMDataBaseSampler(DataBaseSampler):

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 check_2D_collision=False,
                 collision_thr=0,
                 collision_in_classes=False,
                 depth_consistent=False,
                 blending_type=None,
                 mixup=1.0,
                 img_loader=dict(type='LoadImageFromFile'),
                #  mask_loader=dict(
                #      type='LoadImageFromFile', color_type='grayscale'),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     load_dim=4,
                     coord_type='LIDAR',
                     use_dim=[0, 1, 2, 3])):
        super(MMDataBaseSampler, self).__init__(
            info_path=info_path,
            data_root=data_root,
            rate=rate,
            prepare=prepare,
            sample_groups=sample_groups,
            classes=classes,
            points_loader=points_loader)
        
        self.blending_type = blending_type
        self.depth_consistent = depth_consistent
        self.check_2D_collision = check_2D_collision
        self.collision_thr = collision_thr
        self.collision_in_classes = collision_in_classes
        self.img_loader = mmcv.build_from_cfg(img_loader, PIPELINES)
        self.mixup = mixup
        # self.mask_loader = mmcv.build_from_cfg(mask_loader, PIPELINES)

    def sample_all(self, gt_bboxes_3d, gt_names, gt_bboxes_2d=None, img=None, img_filename=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes_3d = []
        sampled_gt_bboxes_2d = []
        avoid_coll_boxes_3d = gt_bboxes_3d
        avoid_coll_boxes_2d = gt_bboxes_2d

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes_3d,
                                                   avoid_coll_boxes_2d)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box_3d = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                        sampled_gt_box_2d = sampled_cls[0]['box2d_camera'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box_3d = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                        sampled_gt_box_2d = np.stack(
                            [s['box2d_camera'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes_3d += [sampled_gt_box_3d]
                    sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                    if self.collision_in_classes:
                        # TODO: check whether check collision check among
                        # classes is necessary
                        avoid_coll_boxes_3d = np.concatenate(
                            [avoid_coll_boxes_3d, sampled_gt_box_3d], axis=0)
                        avoid_coll_boxes_2d = np.concatenate(
                            [avoid_coll_boxes_2d, sampled_gt_box_2d], axis=0)

        ret = None
        origin_img = img.copy()
        if len(sampled) > 0:
            sampled_gt_bboxes_3d = np.concatenate(sampled_gt_bboxes_3d, axis=0)
            sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d, axis=0)

            # Get all objects out along with its label (if original or virtual)
            num_origin = gt_bboxes_2d.shape[0]
            origin_label = np.zeros((num_origin, 1))
            num_virtual = sampled_gt_bboxes_2d.shape[0]
            virtual_label = np.ones((num_virtual, 1))

            all_labels = np.concatenate([origin_label, virtual_label], axis=0)
            all_bboxes_3d = np.concatenate([gt_bboxes_3d, sampled_gt_bboxes_3d], axis=0)
            all_bboxes_2d = np.concatenate([gt_bboxes_2d, sampled_gt_bboxes_2d], axis=0)

            num_obj = all_bboxes_3d.shape[0]
            points_paste_order = np.argsort(all_bboxes_3d[:, 0])
            imgs_paste_order = np.argsort(-all_bboxes_3d[:, 0])
            # first of all, we need to get all points
            point_list = []
            for idx in range(len(sampled)):
                info = sampled[idx]
                pcd_file_path = os.path.join(self.data_root, info['path'])\
                    if self.data_root else info['path']
                results = dict(pts_filename=pcd_file_path)
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])
                point_list.append(s_points)
            sampled_points = point_list[0].cat(point_list)

            # convert all points into range perspective
            # x = points[:, 0]; y = points[:, 1]; z = points[:, 2]
            # r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
            # theta = np.arctan(y / x)
            # phi = np.arccos(z / r)
            
            # filter points in a near-to-far manner
            # for idx in range(num_obj):
            #     inds = np.where(points_paste_order == idx)[0][0]
            #     bbox_2d = all_bboxes_2d[inds]; bbox_3d = all_bboxes_3d[inds]
            #     label = all_labels[inds]
                
            #     # original object
            #     if bbox_2d[-1] == 0:
            #         # get the point cloud range cordinate
            #         bbox_instance = LiDARInstance3DBoxes(bbox_3d)
            #         corners = bbox_instance.corners     # shape (8, 3)
            #         box_x = corners[:, 0]; box_y = corners[:, 1]; box_z = corners[:, 2]
            #         box_r = (box_x ** 2 + box_y ** 2 + box_z ** 2) ** 0.5
            #         box_theta = np.arctan(box_y / box_x)
            #         box_phi = np.arccos(box_z / box_r)
            #         min_theta = np.min(box_theta); max_theta = np.max(box_theta)
            #         min_phi = np.min(box_phi); max_phi = np.max(box_phi)
            #         select_idx = x >= (bbox_3d[0] + 1.5)        # NOTE what to filter?
            #         invalid_theta_idx = np.logigal_and(theta[select_idx] >= min_theta, theta[select_idx] <= max_theta)
            #         invalid_phi_idx = np.logigal_and(phi[select_idx] >= min_phi, phi[select_idx] <= max_phi)
            #         invalid_idx = np.logigal_and(invalid_theta_idx, invalid_phi_idx)
            #         # filter points based on index
            #         valid_idx = np.ones((points.shape[0], ), dtype=np.bool)
            #         valid_idx[select_idx][invalid_idx] = 0
            #         points = points[valid_idx]
            #         theta = theta[valid_idx]
            #         phi = phi[valid_idx]

            # paste images in a far to near manner
            for idx in range(num_obj):
                inds = np.where(imgs_paste_order == idx)[0][0]
                bbox_2d = all_bboxes_2d[inds]; bbox_3d = all_bboxes_3d[inds]
                label = all_labels[inds]
                if label == 0:
                    x1, y1, x2, y2 = [int(ii) for ii in bbox_2d]
                    img[x1:x2, y1:y2] = self.mixup * origin_img[x1:x2, y1:y2] + \
                            (1 - self.mixup) * img[x1:x2, y1:y2]
                else:
                    inds -= num_origin
                    info = sampled[inds]
                    pcd_file_path = os.path.join(
                        self.data_root,
                        info['path']) if self.data_root else info['path']
                    img_file_path = pcd_file_path + '.png'
                    patch_results = dict(
                        img_prefix=None, img_info=dict(filename=img_file_path))
                    s_patch = self.img_loader(patch_results)['img']
                    img = self.paste_obj(
                        img,
                        s_patch,
                        # s_mask,
                        bbox_2d=info['box2d_camera'].astype(np.int32))

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled])
            # for each_img in img:
            #     cv2.imwrite('kitti_code/figs/%s.png' % img_file_path.split('/')[-1], each_img)
            # exit()
            ret = dict(
                img=img,
                gt_labels=gt_labels,
                gt_labels_3d=copy.deepcopy(gt_labels),
                gt_bboxes_3d=sampled_gt_bboxes_3d,
                gt_bboxes_2d=sampled_gt_bboxes_2d,
                points=sampled_points,
                group_ids=np.arange(gt_bboxes_3d.shape[0],
                                    gt_bboxes_3d.shape[0] + len(sampled)))

        return ret

    def paste_obj(self, img, obj_img, bbox_2d):
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        # obj_mask = obj_mask[:h, :w]
        obj_img = obj_img[:h, :w]
        obj_mask = np.zeros((h, w))
        margin_h = int(0.05 * h)
        margin_w = int(0.05 * w)
        obj_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1.

        # choose a blend option
        if not self.blending_type:
            blending_op = 'none'

        else:
            blending_choice = np.random.randint(len(self.blending_type))
            blending_op = self.blending_type[blending_choice]

        if blending_op.find('poisson') != -1:
            # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
            # cv2.MIXED_CLONE mixed the texture, thus is not used.
            if blending_op == 'poisson':
                mode = np.random.choice([1, 3], 1)[0]
            elif blending_op == 'poisson_normal':
                mode = cv2.NORMAL_CLONE
            elif blending_op == 'poisson_transfer':
                mode = cv2.MONOCHROME_TRANSFER
            else:
                raise NotImplementedError
            center = (int(x1 + w / 2), int(y1 + h / 2))
            img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
        else:
            if blending_op == 'gaussian':
                obj_mask = cv2.GaussianBlur(
                    obj_mask.astype(np.float32), (5, 5), 2)
            elif blending_op == 'box':
                obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
            paste_mask = 1 - obj_mask * self.mixup
            img[y1:y1 + h,
                x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                              paste_mask[..., None]).astype(np.uint8)
            img[y1:y1 + h, x1:x1 + w] += (self.mixup * obj_img.astype(np.float32) *
                                          obj_mask[..., None]).astype(np.uint8)

        return img

    def sample_class_v2(self, name, num, gt_bboxes_3d, gt_bboxes_2d):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)
        # avoid collision in BEV first
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6])
        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])
        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        # Then avoid collision in 2D space
        if self.check_2D_collision:
            sp_boxes_2d = np.stack([i['box2d_camera'] for i in sampled],
                                   axis=0)
            total_bbox_2d = np.concatenate([gt_bboxes_2d, sp_boxes_2d],
                                           axis=0)  # Nx4
            # random select a collision threshold
            if isinstance(self.collision_thr, float):
                collision_thr = self.collision_thr
            elif isinstance(self.collision_thr, list):
                collision_thr = np.random.choice(self.collision_thr)
            elif isinstance(self.collision_thr, dict):
                mode = self.collision_thr.get('mode', 'value')
                if mode == 'value':
                    collision_thr = np.random.choice(
                        self.collision_thr['thr_range'])
                elif mode == 'range':
                    collision_thr = np.random.uniform(
                        self.collision_thr['thr_range'][0],
                        self.collision_thr['thr_range'][1])

            if collision_thr == 0:
                # use similar collision test as BEV did
                # Nx4 (x1, y1, x2, y2) -> corners: Nx4x2
                # ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                x1y1 = total_bbox_2d[:, :2]
                x2y2 = total_bbox_2d[:, 2:]
                x1y2 = np.stack([total_bbox_2d[:, 0], total_bbox_2d[:, 3]],
                                axis=-1)
                x2y1 = np.stack([total_bbox_2d[:, 2], total_bbox_2d[:, 1]],
                                axis=-1)
                total_2d = np.stack([x1y1, x2y1, x1y2, x2y2], axis=1)
                coll_mat_2d = data_augment_utils.box_collision_test(
                    total_2d, total_2d)
            else:
                # use iof rather than iou to protect the foreground
                overlaps = box_np_ops.iou_jit(total_bbox_2d, total_bbox_2d,
                                              'iof')
                coll_mat_2d = overlaps > collision_thr
            coll_mat = coll_mat + coll_mat_2d

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples

@OBJECTSAMPLERS.register_module()
class MMDataBaseSamplerV2(DataBaseSampler):

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 check_2D_collision=False,
                 collision_thr=0,
                 collision_in_classes=False,
                 depth_consistent=False,
                 blending_type=None,
                 mixup=1.0,
                 img_num=1,
                 img_loader=dict(type='LoadImageFromFile'),
                #  mask_loader=dict(
                #      type='LoadImageFromFile', color_type='grayscale'),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     load_dim=4,
                     coord_type='LIDAR',
                     use_dim=[0, 1, 2, 3])):
        super(MMDataBaseSamplerV2, self).__init__(
            info_path=info_path,
            data_root=data_root,
            rate=rate,
            prepare=prepare,
            sample_groups=sample_groups,
            classes=classes,
            points_loader=points_loader)
        
        self.blending_type = blending_type
        self.depth_consistent = depth_consistent
        self.check_2D_collision = check_2D_collision
        self.collision_thr = collision_thr
        self.collision_in_classes = collision_in_classes
        self.img_loader = mmcv.build_from_cfg(img_loader, PIPELINES)
        self.mixup = mixup
        self.img_num = img_num
        # self.mask_loader = mmcv.build_from_cfg(mask_loader, PIPELINES)

    def sample_all(self, gt_bboxes_3d, gt_labels, gt_bboxes_2d=None, img=None, img_filename=None):
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes_3d = []
        sampled_gt_bboxes_2d = []
        avoid_coll_boxes_3d = gt_bboxes_3d
        avoid_coll_boxes_2d = gt_bboxes_2d

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes_3d,
                                                   avoid_coll_boxes_2d)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box_3d = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                        sampled_gt_box_2d = sampled_cls[0]['box2d_camera'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box_3d = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                        sampled_gt_box_2d = np.stack(
                            [s['box2d_camera'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes_3d += [sampled_gt_box_3d]
                    sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                    if self.collision_in_classes:
                        # TODO: check whether check collision check among
                        # classes is necessary
                        avoid_coll_boxes_3d = np.concatenate(
                            [avoid_coll_boxes_3d, sampled_gt_box_3d], axis=0)
                        avoid_coll_boxes_2d = np.concatenate(
                            [avoid_coll_boxes_2d, sampled_gt_box_2d], axis=0)

        ret = None
        origin_img = img.copy()
        if len(sampled) > 0:

            # for i in range(len(img)):  # debug
            #     img[i].save('/home/junbo/repository/AutoAlignV2/work_dirs/visual/gt-aug/before_dafqwefrrgsd_{}.jpg'.format(i))

            sampled_gt_bboxes_3d = np.concatenate(sampled_gt_bboxes_3d, axis=0)
            sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d, axis=0)

            # Get all objects out along with its label (if original or virtual)
            num_origin = gt_bboxes_2d.shape[0]
            origin_label = np.zeros((num_origin, 1))
            num_virtual = sampled_gt_bboxes_2d.shape[0]
            virtual_label = np.ones((num_virtual, 1))

            if num_origin == 0:
                all_labels = virtual_label
                all_bboxes_3d = sampled_gt_bboxes_3d
                bboxes_2d = sampled_gt_bboxes_2d
            else:
                all_labels = np.concatenate([origin_label, virtual_label], axis=0)
                all_bboxes_3d = np.concatenate([gt_bboxes_3d, sampled_gt_bboxes_3d], axis=0)
                bboxes_2d = np.concatenate([gt_bboxes_2d, sampled_gt_bboxes_2d], axis=0)

            all_camera_idx = bboxes_2d[:, -1]
            all_bboxes_2d = bboxes_2d[:, :4]

            num_obj = all_bboxes_3d.shape[0]
            imgs_paste_order = np.argsort(-all_bboxes_3d[:, 0])
            # first of all, we need to get all points
            point_list = []
            for idx in range(len(sampled)):
                info = sampled[idx]
                pcd_file_path = os.path.join(self.data_root, info['path'])\
                    if self.data_root else info['path']
                results = dict(pts_filename=pcd_file_path)
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])
                point_list.append(s_points)
            sampled_points = point_list[0].cat(point_list)
            # paste images in a far to near manner
            for idx in range(num_obj):
                inds = np.where(imgs_paste_order == idx)[0][0]
                bbox_2d = all_bboxes_2d[inds]; bbox_3d = all_bboxes_3d[inds]
                camera_idx = int(all_camera_idx[inds])
                # use only one image
                if camera_idx >= self.img_num:
                    continue

                label = all_labels[inds]

                # original
                # if label == 0:
                #     x1, y1, x2, y2 = [int(ii) for ii in bbox_2d]
                #     img[camera_idx][x1:x2, y1:y2] = self.mixup * origin_img[camera_idx][x1:x2, y1:y2] + \
                #             (1 - self.mixup) * img[camera_idx][x1:x2, y1:y2]
                # else:
                #     inds -= num_origin
                #     info = sampled[inds]
                #     pcd_file_path = os.path.join(
                #         self.data_root,
                #         info['path']) if self.data_root else info['path']
                #     img_file_path = pcd_file_path + '.png'
                #     patch_results = dict(
                #         img_prefix=None, img_info=dict(filename=img_file_path))
                #     s_patch = self.img_loader(patch_results)['img']
                #     img[camera_idx] = self.paste_obj(
                #         img[camera_idx],
                #         s_patch,
                #         # s_mask,
                #         bbox_2d=bbox_2d.astype(np.int32))

                if label == 0:  # real GT
                    x1, y1, x2, y2 = [int(ii) for ii in bbox_2d]
                    img[camera_idx] = np.array(img[camera_idx])
                    origin_img[camera_idx] = np.array(origin_img[camera_idx])
                    img[camera_idx][x1:x2, y1:y2] = self.mixup * origin_img[camera_idx][x1:x2, y1:y2] + \
                            (1 - self.mixup) * img[camera_idx][x1:x2, y1:y2]
                    img[camera_idx] = Image.fromarray(img[camera_idx])
                else:   #  paste GT
                    inds -= num_origin
                    info = sampled[inds]
                    pcd_file_path = os.path.join(
                        self.data_root,
                        info['path']) if self.data_root else info['path']
                    img_file_path = pcd_file_path + '.png'
                    patch_results = dict(
                        img_prefix=None, img_info=dict(filename=img_file_path))
                    s_patch = self.img_loader(patch_results)['img']
                    img[camera_idx] = self.paste_obj_v2(
                        img[camera_idx],
                        s_patch,
                        # s_mask,
                        bbox_2d=bbox_2d.astype(np.int32))

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled], dtype=np.long)
            # for camera_idx, each_img in enumerate(img):
            #     cv2.imwrite('nus_code/figs/%s' % img_filename[camera_idx].split('/')[-1], each_img)
            # exit()
            ret = dict(
                img=img,
                gt_labels=gt_labels,
                gt_labels_3d=copy.deepcopy(gt_labels),
                gt_bboxes_3d=sampled_gt_bboxes_3d,
                gt_bboxes_2d=sampled_gt_bboxes_2d,
                points=sampled_points,
                group_ids=np.arange(gt_bboxes_3d.shape[0],
                                    gt_bboxes_3d.shape[0] + len(sampled)))

            # for i in range(len(img)):
            #     img[i].save('/home/junbo/repository/AutoAlignV2/work_dirs/visual/gt-aug/after_dafqwefrrgsd_{}.jpg'.format(i))

        return ret

    def paste_obj(self, img, obj_img, bbox_2d):
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        # obj_mask = obj_mask[:h, :w]
        obj_img = obj_img[:h, :w]
        obj_mask = np.zeros((h, w))
        margin_h = int(0.05 * h)
        margin_w = int(0.05 * w)
        obj_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1.

        # choose a blend option
        if not self.blending_type:
            blending_op = 'none'

        else:
            blending_choice = np.random.randint(len(self.blending_type))
            blending_op = self.blending_type[blending_choice]

        if blending_op.find('poisson') != -1:
            # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
            # cv2.MIXED_CLONE mixed the texture, thus is not used.
            if blending_op == 'poisson':
                mode = np.random.choice([1, 3], 1)[0]
            elif blending_op == 'poisson_normal':
                mode = cv2.NORMAL_CLONE
            elif blending_op == 'poisson_transfer':
                mode = cv2.MONOCHROME_TRANSFER
            else:
                raise NotImplementedError
            center = (int(x1 + w / 2), int(y1 + h / 2))
            img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
        else:
            if blending_op == 'gaussian':
                obj_mask = cv2.GaussianBlur(
                    obj_mask.astype(np.float32), (5, 5), 2)
            elif blending_op == 'box':
                obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
            paste_mask = 1 - obj_mask * self.mixup
            img[y1:y1 + h,
                x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                              paste_mask[..., None]).astype(np.uint8)
            img[y1:y1 + h, x1:x1 + w] += (self.mixup * obj_img.astype(np.float32) *
                                          obj_mask[..., None]).astype(np.uint8)

        return img

    def paste_obj_v2(self, img, obj_img, bbox_2d):
        # paste the image patch back
        x1, y1, x2, y2 = bbox_2d
        # the bbox might exceed the img size because the img is different
        img_w, img_h = img.size
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        # obj_mask = obj_mask[:h, :w]
        obj_img = obj_img[:h, :w]
        obj_mask = np.zeros((h, w))
        margin_h = int(0.05 * h)
        margin_w = int(0.05 * w)
        obj_mask[margin_h:h-margin_h, margin_w:w-margin_w] = 1.

        paste_mask = 1 - obj_mask * self.mixup

        pmg = np.array(img)
        obj_img = obj_img[..., [2,1,0]]

        pmg[y1:y1 + h,
        x1:x1 + w] = (pmg[y1:y1 + h, x1:x1 + w].astype(np.float32) *
                      paste_mask[..., None]).astype(np.uint8)
        pmg[y1:y1 + h, x1:x1 + w] += (self.mixup * obj_img.astype(np.float32) *
                                          obj_mask[..., None]).astype(np.uint8)
        img = Image.fromarray(pmg)

        return img

    def sample_class_v2(self, name, num, gt_bboxes_3d, gt_bboxes_2d):
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)
        # avoid collision in BEV first
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6])
        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])
        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        # Then avoid collision in 2D space
        if self.check_2D_collision:
            sp_boxes_2d = np.stack([i['box2d_camera'] for i in sampled],
                                   axis=0)
            if gt_bboxes_2d.shape[0] == 0:
                total_bbox_2d = sp_boxes_2d
            else:
                total_bbox_2d = np.concatenate([gt_bboxes_2d, sp_boxes_2d],
                                           axis=0)  # Nx4
            # random select a collision threshold
            if isinstance(self.collision_thr, float):
                collision_thr = self.collision_thr
            elif isinstance(self.collision_thr, list):
                collision_thr = np.random.choice(self.collision_thr)
            elif isinstance(self.collision_thr, dict):
                mode = self.collision_thr.get('mode', 'value')
                if mode == 'value':
                    collision_thr = np.random.choice(
                        self.collision_thr['thr_range'])
                elif mode == 'range':
                    collision_thr = np.random.uniform(
                        self.collision_thr['thr_range'][0],
                        self.collision_thr['thr_range'][1])

            if collision_thr == 0:
                # use similar collision test as BEV did
                # Nx4 (x1, y1, x2, y2) -> corners: Nx4x2
                # ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
                x1y1 = total_bbox_2d[:, :2]
                x2y2 = total_bbox_2d[:, 2:4]
                x1y2 = np.stack([total_bbox_2d[:, 0], total_bbox_2d[:, 3]],
                                axis=-1)
                x2y1 = np.stack([total_bbox_2d[:, 2], total_bbox_2d[:, 1]],
                                axis=-1)
                total_2d = np.stack([x1y1, x2y1, x1y2, x2y2], axis=1)
                coll_mat_2d = data_augment_utils.box_collision_test(
                    total_2d, total_2d)
            else:
                # use iof rather than iou to protect the foreground
                overlaps = box_np_ops.iou_jit(total_bbox_2d, total_bbox_2d,
                                              'iof')
                coll_mat_2d = overlaps > collision_thr
            coll_mat = coll_mat + coll_mat_2d

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples

