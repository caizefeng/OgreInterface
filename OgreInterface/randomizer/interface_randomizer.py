import itertools
import random

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from ase.ga.utilities import closest_distances_generator
from ase.ga.startgenerator import StartGenerator
from ase import Atoms
from ase.data import atomic_numbers
import numpy as np
from scipy.stats import norm

from OgreInterface.surfaces.interface import Interface


class InterfaceRandomizer:
    """ """

    def __init__(self, interface: Interface):
        self._init_interface = interface
        self._init_interface_structure = interface.get_interface(
            orthogonal=True,
            return_atoms=False,
        )
        self._unique_atomic_numbers = self._get_unique_atomic_numbers()
        self._min_bond_lengths = self._get_min_bond_lengths()
        self._init_cart_coords = self._init_interface_structure.cart_coords
        self._interface_z_height = self._init_interface.interface_height

    @property
    def film_atomic_layers(self):
        return self._init_interface.film.atomic_layers

    @property
    def substrate_atomic_layers(self):
        return self._init_interface.substrate.atomic_layers

    def _get_substrate_layers_to_randomize_and_rattle(self):
        total_atomic_layers = self.substrate_atomic_layers
        layers_to_pick = np.arange(
            start=2,
            stop=np.round(0.6 * total_atomic_layers),
        )

        layer_distance_from_interface = []

        for layer in layers_to_pick:
            layer_inds = self._init_interface.get_substrate_layer_indices(
                layer_from_interface=layer,
                atomic_layers=True,
            )
            z_coords = self._init_cart_coords[layer_inds][:, -1]
            avg_z_coords = np.mean(z_coords) - self._interface_z_height
            layer_distance_from_interface.append(avg_z_coords)

        layer_distance_from_interface = np.array(layer_distance_from_interface)
        layer_distance_from_interface -= layer_distance_from_interface.min()
        layer_distance_from_interface /= (
            0.5 * layer_distance_from_interface.max()
        )

        distribution = norm(0, 1)
        probs = distribution.pdf(layer_distance_from_interface)

        random_layer = np.random.choice(layers_to_pick, p=probs)

        print(random_layer)

    def _get_film_layers_to_randomize_and_rattle(self):
        pass

    def _get_unique_atomic_numbers(self):
        return np.unique(self._init_interface_structure.atomic_numbers)

    def _get_min_bond_lengths(self):
        min_bond_lengths = closest_distances_generator(
            atom_numbers=self._unique_atomic_numbers,
            ratio_of_covalent_radii=0.8,
        )

        return min_bond_lengths

    def _get_random_displacements(self, structure: Structure) -> np.ndarray:
        try:
            from hiphive.structure_generation.rattle import mc_rattle
        except ImportError:
            raise "HiPhive needs to be installed `pip install hiphive`"

        atoms = AseAtomsAdaptor().get_atoms(structure=structure)
        d_min = min(list(self._min_bond_lengths.values()))

        displacements = mc_rattle(
            atoms=atoms,
            rattle_std=0.5,
            active_atoms=inds_to_rattle,
            d_min=d_min,
        )

    def _get_composition(self, natoms):
        elements = self.random_comp

        compositions = list(
            itertools.combinations_with_replacement(elements, natoms)
        )
        compositions = [
            comp for comp in compositions if all(e in comp for e in elements)
        ]

        ind = random.randint(0, len(compositions) - 1)
        composition = compositions[ind]

        return composition

    def _generate_random_structure(self):
        blocks = self._get_composition(self.natoms)
        unique_e, counts = np.unique(blocks, return_counts=True)

        blmin = closest_distances_generator(
            atom_numbers=[atomic_numbers[i] for i in unique_e],
            ratio_of_covalent_radii=0.9,
        )

        cell = Atoms(cell=np.eye(3) * self.cell_size, pbc=True)

        sg = StartGenerator(
            cell,
            blocks,
            blmin,
            number_of_variable_cell_vectors=0,
        )

        a = sg.get_new_candidate()
        s = AseAtomsAdaptor().get_structure(a)

        return s
