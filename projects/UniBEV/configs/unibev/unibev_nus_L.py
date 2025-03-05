# End-to-end training/test for UniBEV_L
# CNW
# Encoder Dimension: 256
# Decoder Dimension: 256
default_scope = "mmdet3d"
custom_imports = dict(imports=["projects.UniBEV.unibev_plugin"], allow_failed_imports=False)

eval_interval = 1
samples_per_gpu = 1
workers_per_gpu = 2
max_epochs = 36
save_interval = 6
log_interval = 10
fusion_method = "linear"


dataset_type = "NuScenesDataset"
data_root = "data/nuscenes/"
# sub_dir = 'mmdet3d_old_cor/'
train_ann_file = "nuscenes_infos_train.pkl"
val_ann_file = "nuscenes_infos_val.pkl"
work_dir = "./outputs/train/unibev_nus_L_full"

load_from = None

resume_from = None
plugin = True
plugin_dir = "mmdet3d/unibev_plugin/"

## nuscenes and pointpillars setting
point_cloud_range = [-54, -54, -5, 54, 54, 3]
voxel_size = [0.075, 0.075, 0.2]
class_names = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]
file_client_args = dict(backend="disk")
input_modality = dict(use_lidar=True, use_camera=False, use_radar=False, use_map=False, use_external=False)

### model settings
img_scale = (1600, 900)
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
if fusion_method == "linear":
    dec_scale_factor = 1
elif fusion_method == "avg":
    dec_scale_factor = 1
elif fusion_method == "cat":
    dec_scale_factor = 2

_encoder_layers_ = 3
_num_levels_ = 1
_num_points_in_pillar_lidar_ = 4
bev_h_ = 200
bev_w_ = 200

backend_args = None
metainfo = dict(classes=class_names)
num_workers = 8

data_prefix = dict(
    pts="samples/LIDAR_TOP",
    CAM_FRONT="samples/CAM_FRONT",
    CAM_FRONT_LEFT="samples/CAM_FRONT_LEFT",
    CAM_FRONT_RIGHT="samples/CAM_FRONT_RIGHT",
    CAM_BACK="samples/CAM_BACK",
    CAM_BACK_RIGHT="samples/CAM_BACK_RIGHT",
    CAM_BACK_LEFT="samples/CAM_BACK_LEFT",
    sweeps="sweeps/LIDAR_TOP",
)

# runner = dict(type='EpochBasedRunner',
#              max_epochs=max_epochs)
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=5)
val_cfg = dict()
test_cfg = dict()

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        # file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    # dict(type='DefaultFormatBundle3D', class_names=class_names), ## which DefaultFormat
    dict(type="Pack3DDetInputs", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),  ## which data collection
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        # file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # dict(
            #    type='DefaultFormatBundle3D',
            #    class_names=class_names,
            #    with_label=False),
            dict(type="Pack3DDetInputs", keys=["points"])
        ],
    ),
]

eval_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        # file_client_args=file_client_args
    ),
    dict(type="LoadPointsFromMultiSweeps", sweeps_num=10, use_dim=[0, 1, 2, 3, 4], file_client_args=file_client_args),
    # dict(
    #    type='DefaultFormatBundle3D',
    #    class_names=class_names,
    #    with_label=False),
    dict(type="Pack3DDetInputs", keys=["points"]),
]

train_dataloader = dict(
    batch_size=samples_per_gpu,
    num_workers=workers_per_gpu,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        modality=input_modality,
        backend_args=backend_args,
        data_root=data_root,
        ann_file=train_ann_file,
        metainfo=metainfo,
        # class_names=class_names,
        test_mode=False,
        data_prefix=data_prefix,
        box_type_3d="LiDAR",
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        metainfo=metainfo,
        # class_names=class_names,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        metainfo=metainfo,
        # class_names=class_names,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type="NuScenesMetric",
    data_root=data_root,
    ann_file=data_root + "nuscenes_infos_val.pkl",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator

model = dict(
    type="UniBEV",
    use_grid_mask=True,
    use_lidar=input_modality["use_lidar"],
    use_radar=input_modality["use_radar"],
    use_camera=input_modality["use_camera"],
    # pts_voxel_layer=dict(
    #    max_num_points=10,
    #    voxel_size=voxel_size,
    #    point_cloud_range=point_cloud_range,
    #    max_voxels=(90000, 120000)),
    data_preprocessor=dict(
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(90000, 120000),
        ),
    ),
    pts_voxel_encoder=dict(type="HardSimpleVFE", num_features=5),
    pts_middle_encoder=dict(
        type="SparseEncoder",
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=("conv", "norm", "act"),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type="basicblock",
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[_dim_ // 2, _dim_ // 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="UniBEV_Head",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type="UniBEVTransformer",
            embed_dims=_dim_,
            fusion_method=fusion_method,
            pts_encoder=dict(
                type="PtsEncoder",
                num_layers=_encoder_layers_,
                pc_range=point_cloud_range,
                num_points_in_pillar_lidar=_num_points_in_pillar_lidar_,
                return_intermediate=False,
                transformerlayers=dict(
                    type="PtsLayer",
                    attn_cfgs=[
                        dict(type="MultiScaleDeformableAttention", embed_dims=_dim_, num_levels=1),
                        dict(
                            type="SpatialCrossAttentionPts",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3DPts",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=_dim_,
                    ),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="mmdet.MultiheadAttention",
                            embed_dims=_dim_ * dec_scale_factor,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(type="CustomMSDeformableAttention", embed_dims=_dim_ * dec_scale_factor, num_levels=1),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=_dim_ * dec_scale_factor,
                    ),
                    feedforward_channels=_ffn_dim_ * dec_scale_factor,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="mmdet.LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type="mmdet.L1Loss", loss_weight=0.25),
        loss_iou=dict(type="mmdet.GIoULoss", loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            assigner=dict(
                type="HungarianAssigner3DBEVFormer",
                cls_cost=dict(type="mmdet.FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1CostBEVFormer", weight=0.25),
                iou_cost=dict(
                    type="mmdet.IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            )
        )
    ),
)

# evaluation = dict(interval=eval_interval, pipeline=test_pipeline)
# optimizer = dict(
#    type='AdamW',
#    lr=2e-4,
#    paramwise_cfg=dict(
#        custom_keys={
#            'pts_backbone': dict(lr_mult=0.1),
#        }),
#    weight_decay=0.01)
# max_norm=10 is better for SECOND
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(
#    policy='CosineAnnealing',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=1.0 / 3,
#    min_lr_ratio=1e-3)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
            "pts_backbone": dict(lr_mult=0.1),
        },
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)


param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-3,
        T_max=max_epochs,
        begin=0,
        end=max_epochs,
        by_epoch=True,
        # warmup='linear',
        # warmup_ratio=1.0 / 3,
        # warmup_iters=500,
        # warmup_by_epoch=False
    )
]

# runtime settings
""" total_epochs = max_epochs

checkpoint_config = dict(interval=save_interval)
log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
#        dict(type='WandbLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
custom_hooks = [
    dict(type='CheckpointLateStageHook',
         start=21,
         priority=60)
]
workflow = [('train', 1), ('val', 1)] """

auto_scale_lr = dict(enable=False, base_batch_size=8)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=1),
)
