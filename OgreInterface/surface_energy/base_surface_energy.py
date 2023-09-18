from OgreInterface.surfaces import Interface, Surface
from OgreInterface.score_function.generate_inputs import (
    generate_input_dict,
    create_batch,
)
from OgreInterface import utils
from typing import List
import numpy as np
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon
from scipy.interpolate import RectBivariateSpline, CubicSpline
from copy import deepcopy
import torch
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SymmOp, SpacegroupAnalyzer
from pymatgen.io.vasp.inputs import Poscar
from ase.data import chemical_symbols
import itertools
import time
from pyswarms.single.global_best import GlobalBestPSO
import os


class BaseSurfaceEnergy:
    """Base Class for all other surface energy classes

    The BaseSurfaceEnergy contains all the basic methods to perform surface energy calculations
    that other classes can inherit. This class should not be called on it's own, rather it
    should be used as a building block for other surface matching classes

    Args:
        surface: The Surface object generated using the SurfaceGenerator
    """

    def __init__(
        self,
        surface: Surface,
    ):
        self.surface = surface
        self.obs = self.surface.oriented_bulk_structure
        self.slab = utils.get_layer_supercelll(
            structure=self.obs, layers=self.surface.layers
        )

    def _generate_base_inputs(
        self,
        structure: Structure,
        is_slab: bool = True,
    ):
        inputs = generate_input_dict(
            structure=structure,
            cutoff=self._cutoff + 5.0,
            interface=is_slab,
        )

        return inputs

    def get_surface_energy(
        self,
    ):
        """This function calculates the surface energy of the Surface

        Returns:
            Surface energy
        """
        obs_inputs = create_batch(self.obs_inputs, batch_size=1)
        slab_inputs = create_batch(self.slab_inputs, batch_size=1)

        (
            obs_total_energy,
            _,
            _,
            _,
            _,
        ) = self._calculate(obs_inputs)

        (
            slab_total_energy,
            _,
            _,
            _,
            _,
        ) = self._calculate(slab_inputs)

        surface_energy = (
            slab_total_energy - (self.surface.layers * obs_total_energy)
        ) / (2 * self.surface.area)

        return surface_energy[0]
