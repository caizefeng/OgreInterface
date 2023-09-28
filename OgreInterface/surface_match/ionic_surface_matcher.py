from typing import List, Dict, Tuple
from itertools import groupby, combinations_with_replacement, product
from os.path import join, dirname, split

from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import chemical_symbols, covalent_radii
from matscipy.neighbours import neighbour_list
import pandas as pd
import numpy as np

from OgreInterface.score_function import (
    create_batch,
    IonicShiftedForcePotential,
)
from OgreInterface.surfaces import Interface
from OgreInterface.surface_match.base_surface_matcher import BaseSurfaceMatcher


DATA_PATH = join(split(dirname(__file__))[0], "data")


class IonicSurfaceMatcher(BaseSurfaceMatcher):
    """Class to perform surface matching between ionic materials

    The IonicSurfaceMatcher class contain various methods to perform surface matching
    specifically tailored towards an interface between two ionic materials.

    Examples:
        Calculating the 2D potential energy surface (PES)
        >>> from OgreInterface.surface_match import IonicSurfaceMatcher
        >>> surface_matcher = IonicSurfaceMatcher(interface=interface) # interface is Interface class
        >>> E_opt = surface_matcher.run_surface_matching(output="PES.png")
        >>> surface_matcher.get_optmized_structure() # Shift the interface to it's optimal position

        Optimizing the interface in 3D using particle swarm optimization
        >>> from OgreInterface.surface_match import IonicSurfaceMatcher
        >>> surface_matcher = IonicSurfaceMatcher(interface=interface) # interface is Interface class
        >>> E_opt = surface_matcher.optimizePSO(z_bounds=[1.0, 5.0], max_iters=150, n_particles=12)
        >>> surface_matcher.get_optmized_structure() # Shift the interface to it's optimal position

    Args:
        interface: The Interface object generated using the InterfaceGenerator
        grid_density: The sampling density of the 2D potential energy surface plot (points/Angstrom)
    """

    def __init__(
        self,
        interface: Interface,
        grid_density: float = 2.5,
        auto_determine_born_n: bool = False,
        born_n: float = 12.0,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
            # use_interface_energy=use_interface_energy,
        )
        self._ionic_radii_df = pd.read_csv(
            join(DATA_PATH, "ionic_radii_data.csv")
        )
        self._auto_determine_born_n = auto_determine_born_n
        self._born_n = born_n
        self._cutoff = 18.0
        self.charge_dict = self._get_charges()
        self.r0_dict = self._get_r0s(
            sub=self.interface.substrate.bulk_structure,
            film=self.interface.film.bulk_structure,
            charge_dict=self.charge_dict,
        )

        self._add_born_ns(self.iface)
        self._add_born_ns(self.sub_sc_part)
        self._add_born_ns(self.film_sc_part)
        self._add_born_ns(self.sub_bulk)
        self._add_born_ns(self.film_bulk)

        self._add_r0s(self.iface)
        self._add_r0s(self.sub_sc_part)
        self._add_r0s(self.film_sc_part)
        self._add_r0s(self.sub_bulk)
        self._add_r0s(self.film_bulk)

        self.d_interface = self.interface.interfacial_distance
        self.opt_xy_shift = np.zeros(2)
        self.opt_d_interface = self.d_interface

        all_iface_inputs = self._generate_base_inputs(
            structure=self.iface,
            is_slab=True,
        )
        self.const_iface_inputs, self.iface_inputs = self._get_iface_parts(
            inputs=all_iface_inputs
        )

        self.sub_sc_inputs = self._generate_base_inputs(
            structure=self.sub_sc_part,
            is_slab=True,
        )
        self.film_sc_inputs = self._generate_base_inputs(
            structure=self.film_sc_part,
            is_slab=True,
        )
        self.sub_bulk_inputs = self._generate_base_inputs(
            structure=self.sub_bulk,
            is_slab=False,
        )
        self.film_bulk_inputs = self._generate_base_inputs(
            structure=self.film_bulk,
            is_slab=False,
        )

        (
            self.film_energy,
            self.sub_energy,
            self.film_bulk_energy,
            self.sub_bulk_energy,
            self.film_surface_energy,
            self.sub_surface_energy,
            self.const_iface_energy,
        ) = self._get_const_energies()

    def _get_iface_parts(
        self,
        inputs: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        film_film_mask = (
            inputs["is_film"][inputs["idx_i"]]
            & inputs["is_film"][inputs["idx_j"]]
        )
        sub_sub_mask = (~inputs["is_film"])[inputs["idx_i"]] & (
            ~inputs["is_film"]
        )[inputs["idx_j"]]

        const_mask = np.logical_or(film_film_mask, sub_sub_mask)
        # const_mask = film_film_mask

        const_inputs = {}
        variable_inputs = {}

        for k, v in inputs.items():
            if "idx" in k or "offsets" in k:
                const_inputs[k] = v[const_mask]
                variable_inputs[k] = v[~const_mask]
            else:
                const_inputs[k] = v
                variable_inputs[k] = v

        return const_inputs, variable_inputs

    def get_optimized_structure(self):
        opt_shift = self.opt_xy_shift

        self.interface.shift_film_inplane(
            x_shift=opt_shift[0], y_shift=opt_shift[1], fractional=True
        )
        self.interface.set_interfacial_distance(
            interfacial_distance=self.opt_d_interface
        )

        self.iface = self.interface.get_interface(orthogonal=True).copy()

        if self.interface._passivated:
            H_inds = np.where(np.array(self.iface.atomic_numbers) == 1)[0]
            self.iface.remove_sites(H_inds)

        self._add_born_ns(self.iface)
        self._add_r0s(self.iface)
        iface_inputs = self._generate_base_inputs(
            structure=self.iface,
            is_slab=True,
        )
        _, self.iface_inputs = self._get_iface_parts(inputs=iface_inputs)

        self.opt_xy_shift[:2] = 0.0
        self.d_interface = self.opt_d_interface

    def _add_r0s(self, struc):
        r0s = []

        for site in struc:
            atomic_number = site.specie.Z
            if bool(site.properties["is_film"]):
                r0s.append(self.r0_dict["film"][atomic_number])
            else:
                r0s.append(self.r0_dict["sub"][atomic_number])

        struc.add_site_property("r0s", r0s)

    def _add_born_ns(self, struc):
        ion_config_to_n_map = {
            "1s1": 0.0,
            "[He]": 5.0,
            "[Ne]": 7.0,
            "[Ar]": 9.0,
            "[Kr]": 10.0,
            "[Xe]": 12.0,
        }
        n_vals = {}

        Zs = np.unique(struc.atomic_numbers)
        for z in Zs:
            element = Element(chemical_symbols[z])
            ion_config = element.electronic_structure.split(".")[0]
            n_val = ion_config_to_n_map[ion_config]
            if self._auto_determine_born_n:
                n_vals[z] = n_val
            else:
                n_vals[z] = self._born_n

        ns = [n_vals[z] for z in struc.atomic_numbers]
        struc.add_site_property("born_ns", ns)

    def _get_charges(self):
        sub = self.interface.substrate.bulk_structure
        film = self.interface.film.bulk_structure

        oxidation_states = {}

        sub_guess = sub.composition.oxi_state_guesses()

        if len(sub_guess) > 0:
            oxidation_states["sub"] = sub_guess[0]
        else:
            unique_atomic_numbers = np.unique(sub.atomic_numbers)
            oxidation_states["sub"] = {
                chemical_symbols[n]: 0 for n in unique_atomic_numbers
            }

        film_guess = film.composition.oxi_state_guesses()

        if len(film_guess) > 0:
            oxidation_states["film"] = film_guess[0]
        else:
            unique_atomic_numbers = np.unique(film.atomic_numbers)
            oxidation_states["film"] = {
                chemical_symbols[n]: 0 for n in unique_atomic_numbers
            }

        return oxidation_states

    def _get_neighborhood_info_pmg(self, struc, charge_dict):
        struc.add_oxidation_state_by_element(charge_dict)
        Zs = np.unique(struc.atomic_numbers)
        combos = combinations_with_replacement(Zs, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []
        ionic_radii_dict = {Z: [] for Z in Zs}

        cnn = CrystalNN(search_cutoff=7.0, cation_anion=False)
        for i, site in enumerate(struc.sites):
            info_dict = cnn.get_nn_info(struc, i)
            for neighbor in info_dict:
                dist = site.distance(
                    neighbor["site"], jimage=neighbor["image"]
                )
                species = tuple(
                    sorted([site.specie.Z, neighbor["site"].specie.Z])
                )
                neighbor_list.append([species, dist])

        sorted_neighbor_list = sorted(neighbor_list, key=lambda x: x[0])
        groups = groupby(sorted_neighbor_list, key=lambda x: x[0])

        for group in groups:
            nn = list(zip(*group[1]))[1]
            neighbor_dict[group[0]] = np.min(nn)

        for n, d in neighbor_dict.items():
            s1 = chemical_symbols[n[0]]
            s2 = chemical_symbols[n[1]]
            c1 = charge_dict[s1]
            c2 = charge_dict[s2]

            try:
                d1 = float(Element(s1).ionic_radii[c1])
            except KeyError:
                # print(
                #     f"No ionic radius available for {s1}, using the atomic radius instead"
                # )
                d1 = float(Element(s1).atomic_radius)

            try:
                d2 = float(Element(s2).ionic_radii[c2])
            except KeyError:
                # print(
                #     f"No ionic radius available for {s2}, using the atomic radius instead"
                # )
                d2 = float(Element(s2).atomic_radius)

            radius_frac = d1 / (d1 + d2)

            if d is None:
                neighbor_dict[n] = d1 + d2
            else:
                r0_1 = radius_frac * d
                r0_2 = (1 - radius_frac) * d
                ionic_radii_dict[n[0]].append(r0_1)
                ionic_radii_dict[n[1]].append(r0_2)

        mean_radius_dict = {k: np.mean(v) for k, v in ionic_radii_dict.items()}

        return mean_radius_dict

    def _get_neighborhood_info(self, struc, charge_dict):
        struc.add_oxidation_state_by_element(charge_dict)
        Zs = np.unique(struc.atomic_numbers)
        combos = combinations_with_replacement(Zs, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []
        ionic_radii_dict = {Z: [] for Z in Zs}
        coordination_dict = {Z: [] for Z in Zs}

        cnn = CrystalNN(search_cutoff=7.0, cation_anion=True)
        for i, site in enumerate(struc.sites):
            info_dict = cnn.get_nn_info(struc, i)
            coordination_dict[site.specie.Z] = len(info_dict)
            for neighbor in info_dict:
                frac_diff = site.frac_coords - neighbor["site"].frac_coords
                dist = np.linalg.norm(
                    struc.lattice.get_cartesian_coords(frac_diff)
                )
                species = tuple(
                    sorted([site.specie.Z, neighbor["site"].specie.Z])
                )
                neighbor_list.append([species, dist])

        sorted_neighbor_list = sorted(neighbor_list, key=lambda x: x[0])
        groups = groupby(sorted_neighbor_list, key=lambda x: x[0])

        for group in groups:
            nn = list(zip(*group[1]))[1]
            neighbor_dict[group[0]] = np.min(nn)

        for n, d in neighbor_dict.items():
            s1 = chemical_symbols[n[0]]
            s2 = chemical_symbols[n[1]]
            c1 = charge_dict[s1]
            c2 = charge_dict[s2]

            z1_df = self._ionic_radii_df[
                (self._ionic_radii_df["Atomic Number"] == n[0])
                & (self._ionic_radii_df["Oxidation State"] == c1)
            ]

            if len(z1_df) > 0:
                z1_coords = z1_df["Coordination Number"].values
                z1_coord_diff = np.abs(z1_coords - coordination_dict[n[0]])
                z1_coord_mask = z1_coord_diff == z1_coord_diff.min()
                z1_radii = z1_df[z1_coord_mask]

                if not pd.isna(z1_radii["Shannon"]).any():
                    d1 = z1_radii["Shannon"].values.mean() / 100
                else:
                    d1 = z1_radii["ML Mean"].values.mean() / 100
            else:
                d1 = covalent_radii[n[0]]

            z2_df = self._ionic_radii_df[
                (self._ionic_radii_df["Atomic Number"] == n[1])
                & (self._ionic_radii_df["Oxidation State"] == c2)
            ]

            if len(z2_df) > 0:
                z2_coords = z2_df["Coordination Number"].values
                z2_coord_diff = np.abs(z2_coords - coordination_dict[n[1]])
                z2_coord_mask = z2_coord_diff == z2_coord_diff.min()
                z2_radii = z2_df[z2_coord_mask]

                if not pd.isna(z2_radii["Shannon"]).any():
                    d2 = z2_radii["Shannon"].values.mean() / 100
                else:
                    d2 = z2_radii["ML Mean"].values.mean() / 100
            else:
                d2 = covalent_radii[n[1]]

            radius_frac = d1 / (d1 + d2)

            if d is None:
                neighbor_dict[n] = d1 + d2
            else:
                r0_1 = radius_frac * d
                r0_2 = (1 - radius_frac) * d
                ionic_radii_dict[n[0]].append(r0_1)
                ionic_radii_dict[n[1]].append(r0_2)

        mean_radius_dict = {k: np.mean(v) for k, v in ionic_radii_dict.items()}

        return mean_radius_dict

    def _get_neighborhood_info_nn(self, struc, charge_dict):
        atoms = AseAtomsAdaptor().get_atoms(struc)
        atomic_numbers = atoms.get_atomic_numbers()
        unique_atomic_numbers = np.unique(atomic_numbers)
        combos = combinations_with_replacement(unique_atomic_numbers, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []
        ionic_radii_dict = {Z: [] for Z in unique_atomic_numbers}

        idx_i, idx_j, dists = neighbour_list(
            "ijd",
            atoms=AseAtomsAdaptor().get_atoms(struc),
            cutoff=7.0,
        )

        neighbor_list = [
            (tuple(sorted([atomic_numbers[i], atomic_numbers[j]])), d)
            for i, j, d in zip(idx_i, idx_j, dists)
        ]

        neighbor_list.sort(key=lambda x: (x[0], x[1]))
        neighbor_groups = groupby(neighbor_list, key=lambda x: x[0])

        neighbor_dict = {}

        for k, group in neighbor_groups:
            min_dist = min([g[1] for g in group])
            neighbor_dict[k] = min_dist

        for n, d in neighbor_dict.items():
            s1 = chemical_symbols[n[0]]
            s2 = chemical_symbols[n[1]]
            c1 = charge_dict[s1]
            c2 = charge_dict[s2]

            try:
                d1 = float(Element(s1).ionic_radii[c1])
            except KeyError:
                # print(
                #     f"No ionic radius available for {s1}, using the atomic radius instead"
                # )
                d1 = float(Element(s1).atomic_radius)

            try:
                d2 = float(Element(s2).ionic_radii[c2])
            except KeyError:
                # print(
                #     f"No ionic radius available for {s2}, using the atomic radius instead"
                # )
                d2 = float(Element(s2).atomic_radius)

            radius_frac = d1 / (d1 + d2)

            if d is None:
                neighbor_dict[n] = d1 + d2
            else:
                r0_1 = radius_frac * d
                r0_2 = (1 - radius_frac) * d
                ionic_radii_dict[n[0]].append(r0_1)
                ionic_radii_dict[n[1]].append(r0_2)

        mean_radius_dict = {k: np.min(v) for k, v in ionic_radii_dict.items()}

        return mean_radius_dict

    def _get_r0s(self, sub, film, charge_dict):
        sub_radii_dict = self._get_neighborhood_info(
            sub,
            self.charge_dict["sub"],
        )
        film_radii_dict = self._get_neighborhood_info(
            film,
            self.charge_dict["film"],
        )

        r0_dict = {"film": film_radii_dict, "sub": sub_radii_dict}

        return r0_dict

    def pso_function(self, x):
        cart_xy = self.get_cart_xy_shifts(x[:, :2])
        z_shift = x[:, -1] - self.d_interface
        shift = np.c_[cart_xy, z_shift]
        batch_inputs = create_batch(
            inputs=self.iface_inputs,
            batch_size=len(x),
        )

        self._add_shifts_to_batch(
            batch_inputs=batch_inputs,
            shifts=shift,
        )

        E, _, _, _, _ = self._calculate(
            inputs=batch_inputs,
            is_interface=True,
        )
        E_adh, E_iface = self._get_interface_energy(total_energies=E)

        return E_iface

    def _get_const_energies(self):
        sub_sc_inputs = create_batch(
            inputs=self.sub_sc_inputs,
            batch_size=1,
        )
        film_sc_inputs = create_batch(
            inputs=self.film_sc_inputs,
            batch_size=1,
        )

        const_iface_inputs = create_batch(
            inputs=self.const_iface_inputs,
            batch_size=1,
        )

        sub_bulk_inputs = create_batch(
            inputs=self.sub_bulk_inputs,
            batch_size=1,
        )
        film_bulk_inputs = create_batch(
            inputs=self.film_bulk_inputs,
            batch_size=1,
        )

        sub_sc_energy, _, _, _, _ = self._calculate(
            sub_sc_inputs,
            is_interface=False,
        )
        film_sc_energy, _, _, _, _ = self._calculate(
            film_sc_inputs,
            is_interface=False,
        )

        (
            const_iface_energy,
            _,
            self.const_born_energy,
            self.const_coulomb_energy,
            _,
        ) = self._calculate(
            const_iface_inputs,
            is_interface=False,
        )

        sub_bulk_energy, _, _, _, _ = self._calculate(
            sub_bulk_inputs,
            is_interface=False,
        )
        film_bulk_energy, _, _, _, _ = self._calculate(
            film_bulk_inputs,
            is_interface=False,
        )

        N_sub_layers = self.interface.substrate.layers
        N_film_layers = self.interface.film.layers
        # N_sub_sc = np.abs(np.linalg.det(self.interface.match.substrate_sl_transform))
        # N_film_sc = np.linalg.det(self.interface.match.film_sl_transform))
        film_bulk_scale = N_film_layers
        sub_bulk_scale = N_sub_layers

        avg_film_surface_energy = (
            film_sc_energy - (film_bulk_scale * film_bulk_energy)
        ) / (2 * self.interface.area)
        avg_sub_surface_energy = (
            sub_sc_energy - (sub_bulk_scale * sub_bulk_energy)
        ) / (2 * self.interface.area)

        return (
            film_sc_energy[0],
            sub_sc_energy[0],
            film_bulk_energy[0],
            sub_bulk_energy[0],
            avg_film_surface_energy[0],
            avg_sub_surface_energy[0],
            const_iface_energy[0],
        )

    def optimizePSO(
        self,
        z_bounds: List[float],
        max_iters: int = 200,
        n_particles: int = 15,
    ) -> float:
        """
        This function will optimize the interface structure in 3D using Particle Swarm Optimization

        Args:
            z_bounds: A list defining the maximum and minumum interfacial distance [min, max]
            max_iters: Maximum number of iterations of the PSO algorithm
            n_particles: Number of particles to use for the swarm (10 - 20 is usually sufficient)

        Returns:
            The optimal value of the negated adhesion energy (smaller is better, negative = stable, positive = unstable)
        """
        opt_score, opt_pos = self._optimizerPSO(
            func=self.pso_function,
            z_bounds=z_bounds,
            max_iters=max_iters,
            n_particles=n_particles,
        )

        opt_cart = self.get_cart_xy_shifts(opt_pos[:2].reshape(1, -1))
        opt_cart = np.c_[opt_cart, np.zeros(1)]
        opt_frac = opt_cart.dot(self.inv_matrix)[0]

        self.opt_xy_shift = opt_frac[:2]
        self.opt_d_interface = opt_pos[-1]

        return opt_score

    def _calculate(self, inputs: Dict, is_interface: bool = True):
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self._cutoff,
        )

        if is_interface:
            outputs = ionic_potential.forward(
                inputs=inputs,
                constant_coulomb_contribution=self.const_coulomb_energy,
                constant_born_contribution=self.const_born_energy,
            )
        else:
            outputs = ionic_potential.forward(
                inputs=inputs,
            )

        return outputs
