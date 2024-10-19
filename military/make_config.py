from mmengine.config import Config

base = "atss_r101_fpn_1x_coco"
cfg = Config.fromfile(f"configs/atss/{base}.py")
# ----------------------------------------------------
width = 640
height = 640

max_epochs = 2

batch_size = 2
num_classes = 1

dataset_type = "VOCDataset"
classes = (
    "radome",
    "parabol",
)
data_root = "/workspace/data/RadarStation"
# -----------------------------------------------------
cfg.model.bbox_head.num_classes = num_classes
# -----------------------------------------------------
cfg.train_pipeline[2]["scale"] = (width, height)
cfg.test_pipeline[1]["scale"] = (width, height)
# -----------------------------------------------------
cfg.train_dataloader.dataset.type = dataset_type
cfg.train_dataloader.dataset.metainfo = dict(classes=classes)
cfg.train_dataloader.dataset.data_root = data_root
# cfg.train_dataloader.dataset.ann_file = (
#     "/workspace/data/RadarStation/train/_annotations.coco.json"
# )
cfg.train_dataloader.dataset.data_prefix = dict(img="train/")
cfg.train_dataloader.batch_size = batch_size
cfg.train_dataloader.dataset.pipeline[2]["scale"] = (width, height)
# -----------------------------------------------------
cfg.val_dataloader.dataset.type = dataset_type
cfg.val_dataloader.dataset.metainfo = dict(classes=classes)
cfg.val_dataloader.dataset.data_root = data_root
# cfg.val_dataloader.dataset.ann_file = (
#     "/workspace/data/RadarStation/valid/_annotations.coco.json"
# )
cfg.val_dataloader.dataset.data_prefix = dict(img="valid/")
cfg.val_dataloader.dataset.pipeline[1]["scale"] = (width, height)
# -----------------------------------------------------
cfg.test_dataloader.dataset.type = dataset_type
cfg.test_dataloader.dataset.metainfo = dict(classes=classes)
cfg.test_dataloader.dataset.data_root = data_root
# cfg.test_dataloader.dataset.ann_file = (
#     "/workspace/data/RadarStation/test/_annotations.coco.json"
# )

cfg.test_dataloader.dataset.data_prefix = dict(img="test/")
cfg.test_dataloader.dataset.pipeline[1]["scale"] = (width, height)

# ------------------------------------------------------
cfg.val_evaluator.type = "CocoMetric"
# cfg.val_evaluator.ann_file = "/workspace/data/RadarStation/valid/_annotations.coco.json"
# cfg.val_evaluator.metric=['segm']

cfg.test_evaluator.type = "CocoMetric"
# cfg.test_evaluator.ann_file = "/workspace/data/RadarStation/test/_annotations.coco.json"
# cfg.test_evaluator.metric=['segm']
# ------------------------------------------------------
cfg.train_cfg.max_epochs = max_epochs
cfg.optim_wrapper.type = "OptimWrapper"
cfg.optim_wrapper.optimizer = dict(
    type="AdamW", lr=0.001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
)
cfg.default_hooks = dict(
    logger=dict(type="LoggerHook", interval=200),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="auto"),
)

# ------------------------------------------------------
# !mkdir -p configs/HuBMAP
config = f"configs/atss/military_{base}_{width}_{height}.py"
with open(config, "w") as f:
    f.write(cfg.pretty_text)

# train
# python tools/train.py configs/atss/military_atss_r101_fpn_1x_coco_640_640.py
