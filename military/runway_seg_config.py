from mmcv import Config

# Dataset settings
dataset_type = "RunwayDataset"
data_root = "/workspace/data/runway"

# Model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)

optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
# fp16 placeholder
fp16 = dict()
# model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
# Config model heads following this: https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/faq.md#how-to-handle-binary-segmentation-task
model = dict(
    type="EncoderDecoder",
    pretrained="open-mmlab://resnet101_v1c",
    backbone=dict(
        type="ResNetV1c",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style="pytorch",
        contract_dilation=True,
    ),
    decode_head=dict(
        type="ASPPHead",
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# Training pipeline

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "Dice"])


# Training settings
# train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=8000)

# Optimizer settings
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005),
)

# Learning rate scheduler settings
param_scheduler = [
    dict(type="PolyLR", eta_min=1e-4, power=0.9, begin=0, end=80000, by_epoch=False)
]

# Runtime settings
default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="SegLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)

log_processor = dict(by_epoch=False)
log_level = "INFO"
# load_from = None
# resume = False

randomness = dict(seed=None, deterministic=False)


cfg = Config.fromfile(
    "configs/deeplabv3/deeplabv3_r101-d8_fp16_512x1024_80k_cityscapes.py"
)

cfg.data_root = data_root
cfg.dataset_type = dataset_type
cfg.default_scope = default_scope
cfg.env_cfg = env_cfg
cfg.log_level = log_level
cfg.log_processor = log_processor
cfg.model = model
cfg.norm_cfg = norm_cfg
cfg.optim_wrapper = optim_wrapper
cfg.param_scheduler = param_scheduler
cfg.randomness = randomness
cfg.vis_backends = vis_backends
cfg.visualizer = visualizer


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images",
        ann_dir="train/labels",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="Resize", img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
            dict(type="RandomCrop", crop_size=(512, 1024), cat_max_ratio=0.75),
            dict(type="RandomFlip", prob=0.5),
            dict(type="PhotoMetricDistortion"),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="Pad", size=(512, 1024), pad_val=0, seg_pad_val=255),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="valid/images",
        ann_dir="valid/labels",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="test/images",
        ann_dir="test/labels",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)

cfg.data = data
cfg.dump("configs/deeplabv3/runway_config.py")
