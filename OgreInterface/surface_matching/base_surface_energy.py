import typing as tp
from abc import ABC, ABCMeta, abstractmethod

from pymatgen.core.structure import Structure
import numpy as np
from scipy.interpolate import CubicSpline

from OgreInterface.surfaces import BaseSurface
from OgreInterface import utils


# class PostInitCaller(type):
#     def __call__(cls, *args, **kwargs):
#         obj = type.__call__(cls, *args, **kwargs)
#         obj.__post_init__()
#         return obj


# class CombinedPostInitCaller(PostInitCaller, ABCMeta):
#     pass


# , metaclass=CombinedPostInitCaller


class BaseSurfaceEnergy(ABC):
    """Base Class for all other surface energy classes

    The BaseSurfaceEnergy contains all the basic methods to perform surface energy calculations
    that other classes can inherit. This class should not be called on it's own, rather it
    should be used as a building block for other surface matching classes

    Args:
        surface: The Surface object generated using the SurfaceGenerator
    """

    def __init__(
        self,
        surface: BaseSurface,
    ):
        self.surface = surface

        # Number of layers of the slab
        self.layers = self.surface.layers

        # Oriented bulk structure
        self.obs = self.surface.oriented_bulk_structure

        # Area of the OBS structure (slab cross section)
        self.area = self.surface.area

        self.vacuum_scale = np.round(
            self.surface.vacuum / self.surface.oriented_bulk.layer_thickness
        ).astype(int)

        # Slab generated from the OBS
        self.slab = utils.get_layer_supercell(
            structure=self.obs,
            layers=self.layers,
            vacuum_scale=self.vacuum_scale,
        )

        # Self interface generated from the OBS
        self.double_slab = utils.get_layer_supercell(
            structure=self.obs,
            layers=2 * self.layers,
            vacuum_scale=self.vacuum_scale,
        )

        # Layer index property
        double_slab_layers = np.array(
            self.double_slab.site_properties["layer_index"]
        )

        # Is film site properties
        is_film = (double_slab_layers >= self.layers).astype(bool)

        # Add the is_film property to the self interface
        self.double_slab.add_site_property(
            "is_film",
            is_film.tolist(),
        )

        # Get the default interfacial distance
        top_sub = self.double_slab.cart_coords[~is_film][:, -1].max()
        bot_film = self.double_slab.cart_coords[is_film][:, -1].min()

        self._default_distance = bot_film - top_sub

    @abstractmethod
    def generate_constant_inputs(
        self,
        structure: Structure,
    ):
        pass

    @abstractmethod
    def generate_interface_inputs(
        self,
        structure: Structure,
        shifts: np.ndarray,
    ):
        pass

    @abstractmethod
    def calculate(self, inputs):
        pass

    def get_cleavage_energy(self):
        """This function calculates the negated adhesion energy of an interface as a function of the interfacial distance

        Args:
            interfacial_distances: numpy array of the interfacial distances that should be calculated
            figsize: Size of the figure in inches (x_size, y_size)
            fontsize: Fontsize of all the plot labels
            output: Output file name
            dpi: Resolution of the figure (dots per inch)
            save_raw_data_file: If you put a valid file path (i.e. anything ending with .npz) then the
                raw data will be saved there. It can be loaded in via data = np.load(save_raw_data_file)
                and the data is: interfacial_distances = data["interfacial_distances"], energies = data["energies"]

        Returns:
            The optimal value of the negated adhesion energy (smaller is better, negative = stable, positive = unstable)
        """
        default_distance = self._default_distance

        interfacial_distances = np.linspace(
            0.5 * default_distance,
            2.0 * default_distance,
            21,
        )

        zeros = np.zeros(len(interfacial_distances))
        shifts = np.c_[zeros, zeros, interfacial_distances - default_distance]

        double_slab_inputs = self.generate_interface_inputs(
            shifts=shifts,
        )
        double_slab_energies = self.calculate(inputs=double_slab_inputs)

        slab_inputs = self.generate_constant_inputs(structure=self.slab)
        slab_energy = self.calculate(inputs=slab_inputs)[0]

        cleavage_energy = (double_slab_energies - (2 * slab_energy)) / (
            2 * self.area
        )

        cs = CubicSpline(interfacial_distances, cleavage_energy)

        interp_x = np.linspace(
            interfacial_distances.min(),
            interfacial_distances.max(),
            201,
        )
        interp_y = cs(interp_x)

        opt_E = -np.min(interp_y)

        return opt_E

    def get_surface_energy(
        self,
    ):
        """This function calculates the surface energy of the Surface

        Returns:
            Surface energy
        """
        obs_inputs = self.generate_constant_inputs(structure=self.obs)
        slab_inputs = self.generate_constant_inputs(structure=self.slab)

        obs_energy = self.calculate(inputs=obs_inputs)[0]
        slab_energy = self.calculate(inputs=slab_inputs)[0]

        surface_energy = (slab_energy - (self.layers * obs_energy)) / (
            2 * self.area
        )

        return surface_energy
