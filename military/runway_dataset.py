from .builder import DATASETS

# from mmseg.datasets import BaseSegDataset

from .custom import CustomDataset


# Define custom dataset
@DATASETS.register_module()
class RunwayDataset(CustomDataset):
    CLASSES = ("background", "runway")

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".tif", seg_map_suffix=".png", **kwargs)
