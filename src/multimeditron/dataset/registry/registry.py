import abc
from typing import Dict, Any
import os
import numpy as np

class ModalityRegistry(abc.ABC):
    @abc.abstractmethod
    def check_sample(self, sample: Dict[str, Any]) -> bool:
        ...

    @abc.abstractmethod
    def get_modality(self, modality: Dict[str, Any]) -> Any:
        ...

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

    def __enter__(self):
        return self

    def registry_type(self) -> str:
        if not hasattr(self, 'registry_type'):
            raise AttributeError(f"Attribute {registry_type} doesn't exist")
        return self.registry_type
            

def get_registry(registry_type: str) -> ModalityRegistry:
    from multimeditron.dataset.registry.fs_registry import FileSystemImageRegistry
    from multimeditron.dataset.registry.hdf5_registry import HDF5ImageRegistry
    from multimeditron.dataset.registry.wids_registry import WIDSImageRegistry

    match registry_type:
        case "hdf5":
            return HDF5ImageRegistry

        case "wids":
            return WIDSImageRegistry

        case _:
            return FileSystemImageRegistry

