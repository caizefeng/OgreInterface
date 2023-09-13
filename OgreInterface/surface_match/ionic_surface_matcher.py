from OgreInterface.score_function.ionic_shifted_force import (
    IonicShiftedForcePotential,
)
from OgreInterface.score_function.generate_inputs import create_batch
from OgreInterface.surfaces import Interface
from OgreInterface.surface_match.base_surface_matcher import BaseSurfaceMatcher
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import chemical_symbols, covalent_radii
from typing import List, Dict, Tuple
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, CubicSpline
from itertools import groupby, combinations_with_replacement, product

# import torch
import time


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
        # use_interface_energy: bool = True,
        auto_determine_born_n: bool = True,
        born_n: float = 12.0,
    ):
        super().__init__(
            interface=interface,
            grid_density=grid_density,
            # use_interface_energy=use_interface_energy,
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
        # self._add_born_ns(self.sub_surface)
        # self._add_born_ns(self.film_surface)
        self._add_born_ns(self.sub_bulk)
        self._add_born_ns(self.film_bulk)
        self._add_r0s(self.iface)
        self._add_r0s(self.sub_sc_part)
        self._add_r0s(self.film_sc_part)
        # self._add_r0s(self.sub_surface)
        # self._add_r0s(self.film_surface)
        self._add_r0s(self.sub_bulk)
        self._add_r0s(self.film_bulk)
        # print(self.sub_part)
        # print(self.film_part)
        # self._add_charges(self.iface)
        # self._add_charges(self.sub_part)
        # self._add_charges(self.film_part)
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

        # self.sub_surface_inputs = self._generate_base_inputs(
        #     structure=self.sub_surface,
        #     is_slab=False,
        # )
        # self._get_pseudo_surface_inputs(
        #     inputs=self.sub_surface_inputs,
        #     is_film=False,
        # )

        # self.film_surface_inputs = self._generate_base_inputs(
        #     structure=self.film_surface,
        #     is_slab=False,
        # )
        # self._get_pseudo_surface_inputs(
        #     inputs=self.film_surface_inputs,
        #     is_film=True,
        # )

        (
            self.film_energy,
            self.sub_energy,
            self.film_bulk_energy,
            self.sub_bulk_energy,
            self.film_surface_energy,
            self.sub_surface_energy,
            self.const_iface_energy,
        ) = self._get_const_energies()
        # print("Sub Total Energy = ", self.sub_energy)
        # print("Film Total Energy = ", self.film_energy)
        # print("Film Surface Energy = ", self.film_surface_energy)
        # print("Sub Surface Energy = ", self.sub_surface_energy)
        # print("")

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

    def _get_pseudo_surface_inputs(
        self,
        inputs: Dict[str, np.ndarray],
        is_film: bool = True,
    ) -> Dict[str, np.ndarray]:
        if is_film:
            mask = inputs["offsets"][:, -1] >= 0.0
        else:
            mask = inputs["offsets"][:, -1] <= 0.0

        for k, v in inputs.items():
            if "idx" in k or "offsets" in k:
                inputs[k] = v[mask]

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

    # def _add_charges(self, struc):
    #     charges = [
    #         self.charge_dict[chemical_symbols[z]] for z in struc.atomic_numbers
    #     ]
    #     struc.add_site_property("charges", charges)

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
        sub_oxidation_state = sub.composition.oxi_state_guesses()[0]
        film_oxidation_state = film.composition.oxi_state_guesses()[0]

        sub_oxidation_state.update(film_oxidation_state)

        return sub_oxidation_state

    def _get_neighborhood_info(self, struc, charge_dict):
        struc.add_oxidation_state_by_element(charge_dict)
        Zs = np.unique(struc.atomic_numbers)
        combos = combinations_with_replacement(Zs, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []
        ionic_radii_dict = {Z: [] for Z in Zs}

        cnn = CrystalNN(search_cutoff=7.0, cation_anion=True)
        for i, site in enumerate(struc.sites):
            info_dict = cnn.get_nn_info(struc, i)
            for neighbor in info_dict:
                dist = site.distance(neighbor["site"])
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
                print(
                    f"No ionic radius available for {s1}, using the atomic radius instead"
                )
                d1 = float(Element(s1).atomic_radius)

            try:
                d2 = float(Element(s2).ionic_radii[c2])
            except KeyError:
                print(
                    f"No ionic radius available for {s2}, using the atomic radius instead"
                )
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

        return neighbor_dict, mean_radius_dict

    def _get_r0s(self, sub, film, charge_dict):
        sub_dict, sub_radii_dict = self._get_neighborhood_info(
            sub, charge_dict
        )
        film_dict, film_radii_dict = self._get_neighborhood_info(
            film, charge_dict
        )

        r0_dict = {"film": film_radii_dict, "sub": sub_radii_dict}

        return r0_dict

        # interface_atomic_numbers = np.unique(
        #     np.concatenate([sub.atomic_numbers, film.atomic_numbers])
        # )

        # ionic_radius_dict = {}

        # for n in interface_atomic_numbers:
        #     element = Element(chemical_symbols[n])

        #     try:
        #         d = float(
        #             element.ionic_radii[charge_dict[chemical_symbols[n]]]
        #         )

        #     except KeyError:
        #         print(
        #             f"No ionic radius available for {chemical_symbols[n]}, using the atomic radius instead"
        #         )
        #         d = float(element.atomic_radius)

        #     ionic_radius_dict[n] = d

        # interface_combos = product(interface_atomic_numbers, repeat=2)
        # for key in interface_combos:
        #     i = key[0]
        #     j = key[1]

        #     has_sub_i = True
        #     has_sub_j = True

        #     has_film_i = True
        #     has_film_j = True

        #     if i in sub_radii_dict:
        #         sub_r0_i = sub_radii_dict[i]
        #     else:
        #         sub_r0_i = ionic_radius_dict[i]
        #         has_sub_i = False

        #     if j in sub_radii_dict:
        #         sub_r0_j = sub_radii_dict[j]
        #     else:
        #         sub_r0_j = ionic_radius_dict[j]
        #         has_sub_j = False

        #     if i in film_radii_dict:
        #         film_r0_i = film_radii_dict[i]
        #     else:
        #         film_r0_i = ionic_radius_dict[i]
        #         has_film_i = False

        #     if j in film_radii_dict:
        #         film_r0_j = film_radii_dict[j]
        #     else:
        #         film_r0_j = ionic_radius_dict[j]
        #         has_film_j = False

        #     iface_i = ((sub_r0_i * has_sub_i) + (film_r0_i * has_film_i)) / (
        #         has_sub_i + has_film_i
        #     )
        #     iface_j = ((sub_r0_j * has_sub_j) + (film_r0_j * has_film_j)) / (
        #         has_sub_j + has_film_j
        #     )

        #     r0_array[0, i, j] = film_r0_i + film_r0_j
        #     r0_array[1, i, j] = iface_i + iface_j
        #     r0_array[2, i, j] = sub_r0_i + sub_r0_j
        #     r0_array[0, j, i] = film_r0_i + film_r0_j
        #     r0_array[1, j, i] = iface_i + iface_j
        #     r0_array[2, j, i] = sub_r0_i + sub_r0_j

        # return r0_array.astype(np.float32)

    def bo_function(self, a, b, z):
        frac_ab = np.array([a, b]).reshape(1, 2)
        cart_xy = self.get_cart_xy_shifts(frac_ab)
        z_shift = z - self.d_interface
        shift = np.c_[cart_xy, z_shift * np.ones(len(cart_xy))]
        batch_inputs = create_batch(
            inputs=self.iface_inputs,
            batch_size=1,
        )
        E, _, _, _, _ = self._calculate(inputs=batch_inputs, shifts=shift)

        return -E[0]

    def pso_function(self, x):
        cart_xy = self.get_cart_xy_shifts(x[:, :2])
        z_shift = x[:, -1] - self.d_interface
        shift = np.c_[cart_xy, z_shift]
        batch_inputs = create_batch(
            inputs=self.iface_inputs,
            batch_size=len(x),
        )
        E, _, _, _, _ = self._calculate_iface_energy(
            inputs=batch_inputs, shifts=shift
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
            shifts=np.zeros((1, 3)),
        )
        film_sc_energy, _, _, _, _ = self._calculate(
            film_sc_inputs,
            shifts=np.zeros((1, 3)),
        )

        (
            const_iface_energy,
            _,
            self.const_born_energy,
            self.const_coulomb_energy,
            _,
        ) = self._calculate(
            const_iface_inputs,
            shifts=np.zeros((1, 3)),
        )

        print(f"{self.const_born_energy = }")
        print(f"{self.const_coulomb_energy = }")

        sub_bulk_energy, _, _, _, _ = self._calculate(
            sub_bulk_inputs,
            shifts=np.zeros((1, 3)),
        )
        film_bulk_energy, _, _, _, _ = self._calculate(
            film_bulk_inputs,
            shifts=np.zeros((1, 3)),
        )

        # sub_surface_energy, _, _, _, _ = self._calculate(
        #     sub_surface_inputs,
        #     shifts=np.zeros((1, 3)),
        # )
        # film_surface_energy, _, _, _, _ = self._calculate(
        #     film_surface_inputs,
        #     shifts=np.zeros((1, 3)),
        # )
        print(const_iface_energy)

        N_sub_layers = self.interface.substrate.layers
        N_film_layers = self.interface.film.layers
        N_sub_sc = np.linalg.det(self.interface.match.substrate_sl_transform)
        N_film_sc = np.linalg.det(self.interface.match.film_sl_transform)
        film_bulk_scale = N_film_layers * N_film_sc
        sub_bulk_scale = N_sub_layers * N_sub_sc

        # sub_bulk_energy *= sub_bulk_scale
        # film_bulk_energy *= film_bulk_scale

        avg_film_surface_energy = (
            film_sc_energy - (film_bulk_scale * film_bulk_energy)
        ) / (2 * self.interface.area)
        avg_sub_surface_energy = (
            sub_sc_energy - (sub_bulk_scale * sub_bulk_energy)
        ) / (2 * self.interface.area)

        # film_surface_E = (
        #     film_surface_energy - (N_film_layers * film_bulk_energy)
        # ) / (self.interface.film.area)
        # sub_surface_E = (
        #     sub_surface_energy - (N_sub_layers * sub_bulk_energy)
        # ) / (self.interface.substrate.area)

        # print(avg_film_surface_energy, avg_sub_surface_energy)
        # print(film_surface_E, sub_surface_E)

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

    def _calculate(self, inputs: Dict, shifts: np.ndarray):
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self._cutoff,
        )
        outputs = ionic_potential.forward(
            inputs=inputs,
            shift=shifts,
            # r0_array=self.r0_array,
        )

        return outputs

    def _calculate_iface_energy(self, inputs: Dict, shifts: np.ndarray):
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self._cutoff,
        )
        outputs = ionic_potential.forward(
            inputs=inputs,
            shift=shifts,
            constant_coulomb_contribution=self.const_coulomb_energy,
            constant_born_contribution=self.const_born_energy,
            # r0_array=self.r0_array,
        )

        return outputs

    # def _run_bulk(
    #     self,
    #     strains,
    #     fontsize: int = 12,
    #     output: str = "PES.png",
    #     show_born_and_coulomb: bool = False,
    #     dpi: int = 400,
    # ):
    #     # sub = self.interface.substrate.bulk_structure
    #     sub = self.interface.film.bulk_structure
    #     is_film = True

    #     strained_atoms = []
    #     for strain in strains:
    #         strain_struc = sub.copy()
    #         strain_struc.apply_strain(strain)
    #         strain_struc.add_site_property(
    #             "is_film", [is_film] * len(strain_struc)
    #         )
    #         self._add_charges(strain_struc)
    #         self._add_born_ns(strain_struc)
    #         strained_atoms.append(strain_struc)

    #     total_energy = []
    #     coulomb = []
    #     born = []
    #     for i, atoms in enumerate(strained_atoms):
    #         inputs = self._generate_base_inputs(
    #             structure=atoms,
    #         )
    #         batch_inputs = create_batch(inputs, 1)

    #         (
    #             _total_energy,
    #             _coulomb,
    #             _born,
    #             _,
    #             _,
    #         ) = self._calculate(batch_inputs, shifts=np.zeros((1, 3)))
    #         total_energy.append(_total_energy)
    #         coulomb.append(_coulomb)
    #         born.append(_born)

    #     total_energy = np.array(total_energy)
    #     coulomb = np.array(coulomb)
    #     born = np.array(born)

    #     fig, axs = plt.subplots(figsize=(4 * 3, 3), dpi=dpi, ncols=3)
    #     print("Min Strain:", strains[np.argmin(total_energy)])

    #     axs[0].plot(
    #         strains,
    #         total_energy,
    #         color="black",
    #         linewidth=1,
    #         label="Born+Coulomb",
    #     )
    #     axs[1].plot(
    #         strains,
    #         coulomb,
    #         color="red",
    #         linewidth=1,
    #         label="Coulomb",
    #     )
    #     axs[2].plot(
    #         strains,
    #         born,
    #         color="blue",
    #         linewidth=1,
    #         label="Born",
    #     )

    #     for ax in axs:
    #         ax.tick_params(labelsize=fontsize)
    #         ax.set_ylabel("Energy", fontsize=fontsize)
    #         ax.set_xlabel("Strain ($\\AA$)", fontsize=fontsize)
    #         ax.legend(fontsize=12)

    #     fig.tight_layout()
    #     fig.savefig(output, bbox_inches="tight")
    #     plt.close(fig)

    # def _run_scale(
    #     self,
    #     scales,
    #     fontsize: int = 12,
    #     output: str = "scale.png",
    #     show_born_and_coulomb: bool = False,
    #     dpi: int = 400,
    # ):
    #     # sub = self.interface.substrate.bulk_structure
    #     sub = self.interface.film.bulk_structure

    #     strains = np.linspace(-0.1, 0.1, 21)
    #     strained_atoms = []
    #     # for strain in [-0.02, -0.01, 0.0, 0.01, 0.02]:
    #     for strain in strains:
    #         strain_struc = sub.copy()
    #         strain_struc.apply_strain(strain)
    #         strain_atoms = AseAtomsAdaptor().get_atoms(strain_struc)
    #         strain_atoms.set_array(
    #             "is_film", np.zeros(len(strain_atoms)).astype(bool)
    #         )
    #         strained_atoms.append(strain_atoms)

    #     total_energy = []
    #     for scale in scales:
    #         strain_energy = []
    #         for atoms in strained_atoms:
    #             inputs = self._generate_inputs(
    #                 atoms=atoms, shifts=[np.zeros(3)], interface=False
    #             )
    #             ionic_potential = IonicShiftedForcePotential(
    #                 cutoff=self._cutoff,
    #             )
    #             (_total_energy, _, _, _, _,) = ionic_potential.forward(
    #                 inputs=inputs,
    #                 r0_dict=scale * self.r0_array,
    #                 ns_dict=self.ns_dict,
    #                 z_shift=False,
    #             )
    #             strain_energy.append(_total_energy)
    #         total_energy.append(strain_energy)

    #     total_energy = np.array(total_energy)
    #     # coulomb = np.array(coulomb)
    #     # born = np.array(born)

    #     fig, axs = plt.subplots(figsize=(6, 3), dpi=dpi, ncols=2)

    #     colors = plt.cm.jet
    #     color_list = [colors(i) for i in np.linspace(0, 1, len(total_energy))]

    #     min_strains = []
    #     min_Es = []
    #     for i, E in enumerate(total_energy):
    #         E -= E.min()
    #         E /= E.max()
    #         axs[0].plot(
    #             strains,
    #             E,
    #             color=color_list[i],
    #             linewidth=1,
    #             # marker=".",
    #             # alpha=0.3,
    #         )
    #         min_strain = strains[np.argmin(E)]
    #         min_E = E.min()
    #         min_strains.append(min_strain)
    #         min_Es.append(min_E)
    #         axs[0].scatter(
    #             [min_strain],
    #             [min_E],
    #             c=[color_list[i]],
    #             s=2,
    #         )

    #     axs[1].plot(
    #         scales, np.array(min_strains) ** 2, color="black", marker="."
    #     )

    #     fig.tight_layout()
    #     fig.savefig(output, bbox_inches="tight")
    #     plt.close(fig)
