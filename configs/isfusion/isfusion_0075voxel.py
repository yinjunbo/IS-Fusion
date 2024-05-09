class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5, 54, 54, 3]
img_scale = (384, 1056)

total_epochs = 10

res_factor = 1
out_size_factor = 8
voxel_shape = int((point_cloud_range[3]-point_cloud_range[0])//voxel_size[0])
bev_size = voxel_shape//out_size_factor
grid_size = [[bev_size, bev_size, 1], [bev_size//2, bev_size//2, 1]]
region_shape = [(6, 6, 1), (6, 6, 1)]
region_drop_info = [
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
]


model = dict(
    type='ISFusionDetector',

    detach=True,
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    out_size_factor=out_size_factor,

    # img
    img_backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=False,
    ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3),

    # pts
    pts_voxel_layer=dict(
        point_cloud_range=point_cloud_range,
        max_num_points=-1, voxel_size=voxel_size, max_voxels=(-1, -1)),
    pts_voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=5 ,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, voxel_shape, voxel_shape],
        base_channels=32,
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
    ),

    # multi-modal
    fusion_encoder=dict(
        type='ISFusionEncoder',
        num_points_in_pillar=12,
        embed_dims=256,
        bev_size=bev_size,
        num_views=6,
        region_shape=region_shape,
        grid_size=grid_size,
        region_drop_info=region_drop_info,
        instance_num=200,
    ),

    pts_backbone=dict(
        type='SECONDV2',
        in_channels=128,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),

    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    pts_bbox_head = dict(
        type='TransFusionHeadV2',
        num_proposals=200,
        auxiliary=True,
        in_channels=256 * 2,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),


    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[voxel_shape, voxel_shape, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[voxel_shape, voxel_shape, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
            use_rotate_nms=True,  # only for TTA
            nms_thr=0.2,
            max_num=200,
        )))

# If point cloud range is changed, the models should also change their point
# cloud range accordingly

# For nuScenes we usually do 10-class detection
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
# file_client_args = dict(backend='disk')

db_sampler = dict(
    type='MMDataBaseSamplerV2',
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    img_num=6,
    blending_type=None,
    depth_consistent=True,
    check_2D_collision=True,
    collision_thr=[0, 0.3, 0.5, 0.7], #[0, 0.3, 0.5, 0.7],
    # collision_in_classes=True,
    mixup=0.7,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],)
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesV2', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        painting=False),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        painting=False,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type='ObjectSampleV2', stop_epoch=total_epochs-2, db_sampler=db_sampler, sample_2d=True),

    dict(type='ModalMask3D', mode='train', stop_epoch=total_epochs-2,),

    dict(
        type='ImageAug3D',
        final_dim=img_scale,
        resize_lim=[0.57, 0.825],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),

    dict(
        type='GlobalRotScaleTransV2',
        resize_lim=[0.9, 1.1],
        rot_lim=[-0.78539816, 0.78539816],
        trans_lim=0.5,
        is_train=True),

    dict(
        type='RandomFlip3DV2'),

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    dict(
        type='ImageNormalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),

    dict(type='PointShuffle'),

    dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']) # if lidar-only
    dict(type='Collect3DV2', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=[
             'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
             'camera2lidar', 'lidar2img', 'img_aug_matrix', 'lidar_aug_matrix',
         ])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        painting=False,
    ),
    # file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        painting=False,
    ),
    dict(type='LoadMultiViewImageFromFilesV2', to_float32=True),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),

    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1.0,
        flip=False,
        pcd_horizontal_flip=False,
        pcd_vertical_flip=False,
        transforms=[
            dict(
                type='ImageAug3D',
                final_dim=img_scale,
                resize_lim=[0.72, 0.72],
                bot_pct_lim=[0.0, 0.0],
                rot_lim=[0.0, 0.0],
                rand_flip=False,
                is_train=False),
            dict(
                type='ImageNormalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='GlobalRotScaleTransV2',
                resize_lim=[1.0, 1.0],
                rot_lim=[0.0, 0.0],
                trans_lim=0.0,
                is_train=False),
            dict(type='RandomFlip3DV2'), # todo: will work when given annotations, improve it
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3DV2', keys=['points', 'img'],
                 meta_keys=[
                     'camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera',
                     'camera2lidar', 'lidar2img', 'img_aug_matrix', 'lidar_aug_matrix',
                 ])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',
        # type='SimpleDataset',
        # times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=False,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            img_num=6,
            load_interval=1)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        img_num=6,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        img_num=6,
        box_type_3d='LiDAR'))


optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01, paramwise_cfg=dict(
    custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),)  # for 8gpu * 2sample_per_gpu

optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)


# runtime settings
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True, priority='HIGH')]
runner = dict(type='CustomEpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=total_epochs//2)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'data/pretrain_models/swint-nuimages-pretrained-e2e.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
find_unused_parameters=True
