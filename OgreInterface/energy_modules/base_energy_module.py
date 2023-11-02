from abc import ABC, abstractmethod
import typing as tp

from pymatgen.core.structure import Structure
import numpy as np

from OgreInterface.surfaces.base_surface import BaseSurface
from OgreInterface.interfaces.base_interface import BaseInterface


class BaseEnergyModule(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate(self, inputs: tp.Any) -> np.ndarray:
        pass
