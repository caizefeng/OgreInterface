from typing import List, Dict

from pymatgen.core.structure import Structure
import numpy as np
from scipy.interpolate import CubicSpline

from OgreInterface.score_function import (
    generate_input_dict,
    create_batch,
)
from OgreInterface import utils


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
        oriented_bulk_structure: Structure,
        layers: int,
    ):
        self.layers = layers
        self.obs = oriented_bulk_structure
        matrix = self.obs.lattice.matrix
        self.area = np.linalg.norm(np.cross(matrix[0], matrix[1]))
        self.slab = utils.get_layer_supercell(
            structure=self.obs,
            layers=layers,
        )
        self.iface = utils.get_layer_supercell(
            structure=self.obs,
            layers=2 * layers,
        )
        iface_layers = np.array(self.iface.site_properties["layer_index"])
        is_film = (iface_layers >= layers).astype(bool)
        self.iface.add_site_property(
            "is_film",
            is_film.tolist(),
        )
        top_sub = self.iface.cart_coords[~is_film][:, -1].max()
        bot_film = self.iface.cart_coords[is_film][:, -1].min()
        self._default_distance = bot_film - top_sub

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

    def _add_shifts_to_batch(
        self,
        batch_inputs: Dict[str, np.ndarray],
        shifts: np.ndarray,
    ) -> None:
        if "is_film" in batch_inputs:
            n_atoms = batch_inputs["n_atoms"]
            all_shifts = np.repeat(
                shifts.astype(batch_inputs["R"].dtype), repeats=n_atoms, axis=0
            )
            all_shifts[~batch_inputs["is_film"]] *= 0.0
            batch_inputs["R"] += all_shifts
        else:
            raise "_add_shifts_to_batch should only be used on interfaces that have the is_film property"

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

        inputs = create_batch(
            self.iface_inputs,
            batch_size=len(interfacial_distances),
        )
        self._add_shifts_to_batch(batch_inputs=inputs, shifts=shifts)

        (
            interface_energy,
            _,
            _,
            _,
            _,
        ) = self._calculate(inputs)

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

        cleavage_energy = (interface_energy - (2 * slab_total_energy)) / (
            2 * self.area
        )

        cs = CubicSpline(interfacial_distances, cleavage_energy)

        new_x = np.linspace(
            interfacial_distances.min(),
            interfacial_distances.max(),
            201,
        )
        new_y = cs(new_x)
        opt_E = np.min(new_y)

        return -opt_E

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
            slab_total_energy - (self.layers * obs_total_energy)
        ) / (2 * self.area)

        return surface_energy[0]
