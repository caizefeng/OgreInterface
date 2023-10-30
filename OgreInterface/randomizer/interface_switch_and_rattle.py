import itertools
import random
import typing as tp

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from ase.ga.utilities import closest_distances_generator
from ase.ga.startgenerator import StartGenerator
from ase import Atoms
from ase.data import atomic_numbers
import numpy as np
from scipy.stats import norm
from matscipy.neighbours import neighbour_list

from OgreInterface.surfaces.interface import Interface
from OgreInterface import utils


class InterfaceSwitchAndRattile:
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

    def _get_unique_atomic_numbers(self):
        return np.unique(self._init_interface_structure.atomic_numbers)

    def _get_min_bond_lengths(self):
        min_bond_lengths = closest_distances_generator(
            atom_numbers=self._unique_atomic_numbers,
            ratio_of_covalent_radii=0.8,
        )

        return min_bond_lengths

    def _get_neighborhood(
        self,
        atoms: Atoms,
        cutoff: float = 8.0,
    ) -> np.ndarray:
        idx_i, idx_j, d = neighbour_list(
            "ijd",
            atoms=atoms,
            cutoff=cutoff,
        )

        return idx_i, idx_j, d

    def _get_random_displacements(
        self,
        atoms: Atoms,
        seed: int = 42,
        rattle_std: float = 0.05,
    ) -> np.ndarray:
        try:
            from hiphive.structure_generation.rattle import mc_rattle
        except ImportError:
            raise "HiPhive needs to be installed `pip install hiphive`"

        d_min = min(list(self._min_bond_lengths.values()))

        displacements = mc_rattle(
            atoms=atoms,
            rattle_std=rattle_std,
            d_min=d_min,
            seed=seed,
        )

        return displacements

    def randomize_structure(
        self,
        supercell_scaling_matrix: tp.List[int] = [2, 2, 1],
        avg_decay_length: float = 3.0,
        max_flip_distance: float = 6.0,
        substrate_atomic_layers_to_flip: int = 4,
        film_atomic_layers_to_flip: int = 4,
        flip_fraction: float = 0.05,
        seed: int = None,
    ):
        if seed is None:
            seed = np.random.SeedSequence().generate_state(1)[0]

        rng = np.random.Generator(np.random.PCG64(seed=seed))
        inplane_shift = rng.uniform(low=0.0, high=1.0, size=2)
        interfacial_distance = rng.uniform(low=1.75, high=3.5)
        decay_length = np.abs(rng.normal(loc=avg_decay_length, scale=0.5))
        rattle_std = np.abs(rng.normal(loc=0.0, scale=0.075))

        self._init_interface.shift_film_inplane(
            x_shift=inplane_shift[0],
            y_shift=inplane_shift[1],
            fractional=True,
        )
        self._init_interface.set_interfacial_distance(
            interfacial_distance=interfacial_distance
        )

        random_structure = self._init_interface.get_interface(
            orthogonal=True
        ).copy()
        random_structure.make_supercell(scaling_matrix=[2, 2, 1])
        is_film = np.array(random_structure.site_properties["is_film"])

        atoms = AseAtomsAdaptor().get_atoms(structure=random_structure)

        idx_i, idx_j, d_ij = self._get_neighborhood(
            atoms=atoms,
            cutoff=max_flip_distance,
        )

        is_film_i = is_film[idx_i]
        is_film_j = is_film[idx_j]

        film_sub_mask = np.logical_xor(is_film_i, is_film_j)

        idx_i = idx_i[film_sub_mask]
        idx_j = idx_j[film_sub_mask]
        d_ij = d_ij[film_sub_mask]

        possible_flip_inds = []

        for i in range(substrate_atomic_layers_to_flip):
            layer_inds = utils.get_substrate_layer_indices(
                interface=random_structure,
                layer_from_interface=i,
                atomic_layers=True,
            )
            possible_flip_inds.append(layer_inds)

        for i in range(film_atomic_layers_to_flip):
            layer_inds = utils.get_film_layer_indices(
                interface=random_structure,
                layer_from_interface=i,
                atomic_layers=True,
            )
            possible_flip_inds.append(layer_inds)

        possible_flip_inds = np.concatenate(possible_flip_inds)

        flip_i = np.isin(idx_i, possible_flip_inds)
        flip_j = np.isin(idx_j, possible_flip_inds)
        flip_ij = flip_i & flip_j

        idx_i = idx_i[flip_ij]
        idx_j = idx_j[flip_ij]
        d_ij = d_ij[flip_ij]

        sorted_idx_ij = np.sort(np.c_[idx_i, idx_j], axis=1)
        neighbor_info = np.round(np.c_[sorted_idx_ij, d_ij], 4)
        unique_neighbor_info = np.unique(neighbor_info, axis=0)

        unique_tuple = sorted([tuple(i) for i in unique_neighbor_info])
        groups = itertools.groupby(
            unique_tuple, key=lambda x: (int(round(x[0])), int(round(x[1])))
        )

        min_neighbor_info = []

        for ij, group in groups:
            _, _, d = list(zip(*group))
            min_neighbor_info.append([ij[0], ij[1], min(d)])

        min_neighbor_info = np.vstack(min_neighbor_info)
        N_flip_options = len(min_neighbor_info)

        flip_probability = 1 / min_neighbor_info[:, -1]
        flip_probability /= flip_probability.sum()

        inds_to_flip = rng.choice(
            np.arange(N_flip_options),
            size=rng.choice(
                np.arange(int(np.ceil(flip_fraction * len(random_structure))))
            ),
            replace=False,
            p=flip_probability,
        )

        neighbors_to_flip = min_neighbor_info[inds_to_flip]

        for _i, _j, _ in neighbors_to_flip:
            i = int(_i)
            j = int(_j)
            species_i = random_structure[i].species
            species_j = random_structure[j].species

            random_structure[i].species = species_j
            random_structure[j].species = species_i

        atoms = AseAtomsAdaptor().get_atoms(structure=random_structure)

        displacements = self._get_random_displacements(
            atoms=atoms,
            seed=seed,
            rattle_std=rattle_std,
        )

        z_coords = random_structure.cart_coords[:, -1]
        z_dist_from_interface = np.abs(z_coords - self._interface_z_height)
        displacement_scale = 1 - (
            1 / (1 + np.exp(-(z_dist_from_interface - decay_length)))
        )
        displacements *= displacement_scale[:, None]

        rattled_structure = Structure(
            lattice=random_structure.lattice,
            species=random_structure.species,
            coords=random_structure.cart_coords + displacements,
            site_properties=random_structure.site_properties,
            coords_are_cartesian=True,
            to_unit_cell=True,
        )
        rattled_structure.sort()

        return rattled_structure
