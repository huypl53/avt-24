from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


# Define custom dataset
@DATASETS.register_module()
class RunwayDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "runway"), palette=[[0, 0, 0], [255, 255, 255]]
    )

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".tif", seg_map_suffix=".png", **kwargs)
