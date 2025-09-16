import h5py
import os
import numpy as np
from multimeditron.dataset.registry.registry import ModalityRegistry
import PIL
from io import BytesIO
from typing import Dict, Any

class HDF5ImageRegistry(ModalityRegistry):
    registry_type = "hdf5"

    def __init__(self, hdf5_path: str):
        self.h5_path = hdf5_path
    
    def check_sample(self, sample: Dict[str, Any]) -> bool:
        with h5py.File(self.h5_path, "r") as h5_file:
            for modality in sample["modalities"]:
                group = modality["group"]
                paths = h5_file[group]["path"][:]
                search_path = modality["value"]
                
                idx = np.where(paths == search_path.encode("utf-8"))
                if len(idx) == 0:
                    return False
            
            return True

    def get_modality(self, modality: Dict[str, Any]) -> np.ndarray:
        with h5py.File(self.h5_path, "r", swmr=True) as h5_file:
            index = np.where(h5_file[group]["path"][:] == key.encode("utf-8"))
            
            if len(index) == 0:
                raise FileNotFoundError("Modality not found. path: {path}, group: {group}")
            
            image = PIL.Image.open(BytesIO(h5_file[group]["bytes"][index[0]]))
        
            return image
