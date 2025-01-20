from mmengine.config import Config

# Dataset settings
# dataset_type = "RunwayDataset"
# data_root = "/workspace/data/runway"

dataset_type = "RunwayDataset"
data_root = "/workspace/data/runway"
# Model settings

optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
# fp16 placeholder
fp16 = dict()

crop_size = (769, 769)
# data_preprocessor = dict(size=crop_size)

# Config model heads following this: https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/faq.md#how-to-handle-binary-segmentation-task


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

randomness = dict(seed=None, deterministic=False)


### new added ###

# cityscapes.py
# dataset settings
dataset_type = "RunwayDataset"
data_root = "/workspace/data/runway"
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]


# deeplabv3_r50-d8.py
# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)

# cityscapes_769x769.py
# _base_ = './cityscapes.py'
crop_size = (769, 769)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomResize", scale=(2049, 1025), ratio_range=(0.5, 2.0), keep_ratio=True
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2049, 1025), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="train/images", seg_map_path="train/labels"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="valid/images", seg_map_path="valid/labels"),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

# default_runtime.py
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
load_from = None
resume = False

tta_model = dict(type="SegTTAModel")

# schedule_80k.py
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)
# training schedule for 80k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=8000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=8000),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)


# deeplabv3_r50-d8_4xb2-80k_cityscapes-769x769.py

crop_size = (769, 769)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained="open-mmlab://resnet18_v1c",
    backbone=dict(
        type="ResNetV1c",
        depth=18,
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
        in_channels=512,
        in_index=3,
        channels=128,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        # in_channels=1024,
        in_index=2,
        # channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        # align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
        # align_corners=True,
        # in_channels=512,
        # channels=128,
        align_corners=True,
        in_channels=256,
        channels=64,
    ),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode="whole"),
    # decode_head=dict(),
    test_cfg=dict(mode="slide", crop_size=(769, 769), stride=(513, 513)),
    # decode_head=dict(
    # ),
    # auxiliary_head=dict(),
)

# deeplabv3_r18-d8_4xb2-80k_cityscapes-769x769.py

cfg = Config.fromfile(
    "configs/deeplabv3/deeplabv3_r18-d8_4xb2-80k_cityscapes-769x769.py"
)

cfg.dataset_type = dataset_type
cfg.data_root = data_root
cfg.optimizer_config = optimizer_config
cfg.fp16 = fp16
cfg.crop_size = crop_size
cfg.param_scheduler = param_scheduler
cfg.default_scope = default_scope
cfg.env_cfg = env_cfg
cfg.randomness = randomness
cfg.img_ratios = img_ratios
cfg.tta_pipeline = tta_pipeline
cfg.norm_cfg = norm_cfg
cfg.train_pipeline = train_pipeline
cfg.test_pipeline = test_pipeline
cfg.train_dataloader = train_dataloader
cfg.val_dataloader = val_dataloader
cfg.test_dataloader = test_dataloader
cfg.val_evaluator = val_evaluator
cfg.test_evaluator = test_evaluator
cfg.vis_backends = vis_backends
cfg.visualizer = visualizer
cfg.log_processor = log_processor
cfg.log_level = log_level
cfg.load_from = load_from
cfg.resume = resume
cfg.tta_model = tta_model
cfg.optimizer = optimizer
cfg.optim_wrapper = optim_wrapper
cfg.train_cfg = train_cfg
cfg.val_cfg = val_cfg
cfg.test_cfg = test_cfg
cfg.data_preprocessor = data_preprocessor
cfg.model = model
# cfg.data = data
cfg.dump("configs/deeplabv3/runway_config.py")
