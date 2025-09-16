import os
from typing import Dict, Any
from multimeditron.dataset.registry.registry import ModalityRegistry
import numpy as np
import PIL

import warnings

warnings.simplefilter("error", PIL.Image.DecompressionBombWarning)

class FileSystemImageRegistry(ModalityRegistry):
    registry_type = "fs"

    def __init__(self, base_path):
        self.base_path = base_path

    def get_modality(self, modality: Dict[str, Any]) -> np.ndarray:
        image_path = os.path.join(self.base_path, modality["value"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")

        # Load png/jpg/jpeg images
        image = PIL.Image.open(image_path).convert("RGB")
        return image

    def check_sample(self, sample: Dict[str, Any]) -> bool:
        for modality in sample["modalities"]:
            path = os.path.join(self.base_path, modality["value"])
            if not os.path.exists(path):
                return False

            try:
                image = np.array(PIL.Image.open(path).convert("RGB"))
                if image.shape[0] <= 1:
                    del image
                    return False
            except:
                return False
        
        return True


