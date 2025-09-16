import os
import numpy as np
from multimeditron.dataset.registry.registry import ModalityRegistry
import warnings
try:
    import wids
except ImportError:
    warnings.warn("wids package not found. Please install it via `pip install wids`.")

import PIL
from io import BytesIO
import numpy as np
from typing import Dict, Any


class WIDSImageRegistry(ModalityRegistry):
    registry_type = "wids"

    def __init__(self, index_path: str, cache_path: str):
        self.index_path = index_path
        self.cache_path = cache_path

    def check_sample(self, sample: Dict[str, Any]) -> bool:
        for modality in sample["modalities"]:
            wids_index = modality["wds_index"]
            try:
                image_bytes = self.image_dataset[wids_index][".image_bytes"]
            except:
                return False
        return True


    def get_modality(self, modality: Dict[str, Any]) -> np.ndarray:
        wids_index = modality["wds_index"]
        image = PIL.Image.open(self.image_dataset[wids_index][".image_bytes"])
        return image

    def __enter__(self):
        self.image_dataset = wids.ShardListDataset(self.index_path, cache_dir=self.cache_path)
        return self


