from typing import List, Dict, Tuple, Optional
from itertools import groupby, combinations_with_replacement, product
from os.path import join, dirname, split, abspath

from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from matscipy.neighbours import neighbour_list
from ase.data import chemical_symbols, covalent_radii
import pandas as pd
import numpy as np

from OgreInterface import data
from OgreInterface.surface_energy.base_surface_energy import BaseSurfaceEnergy


DATA_PATH = dirname(abspath(data.__file__))


class IonicSurfaceEnergy(BaseSurfaceEnergy):
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
        oriented_bulk_structure: Structure,
        layers: int,
        auto_determine_born_n: bool = False,
        born_n: float = 12.0,
        charge_dict: Optional[Dict[str, float]] = None,
        r0_dict: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            oriented_bulk_structure=oriented_bulk_structure,
            layers=layers,
        )
        self._ionic_radii_df = pd.read_csv(
            join(DATA_PATH, "ionic_radii_data.csv")
        )
        self._auto_determine_born_n = auto_determine_born_n
        self._born_n = born_n
        self._cutoff = 18.0

        if charge_dict is None:
            self.charge_dict = self._get_charges()
        else:
            self.charge_dict = charge_dict

        if r0_dict is None:
            self.r0_dict = self._get_r0s(
                bulk=self.obs,
                charge_dict=self.charge_dict,
            )
        else:
            self.r0_dict = r0_dict

        self._add_born_ns(self.obs)
        self._add_born_ns(self.slab)
        self._add_born_ns(self.iface)
        self._add_r0s(self.obs)
        self._add_r0s(self.slab)
        self._add_r0s(self.iface)

        self.obs_inputs = self._generate_base_inputs(
            structure=self.obs,
            is_slab=False,
        )
        self.slab_inputs = self._generate_base_inputs(
            structure=self.slab,
            is_slab=True,
        )
        self.iface_inputs = self._generate_base_inputs(
            structure=self.iface,
            is_slab=True,
        )

    def _add_r0s(self, struc):
        r0s = []

        for site in struc:
            # atomic_number = site.specie.Z
            atomic_number = site.properties["bulk_equivalent"]
            r0s.append(self.r0_dict[atomic_number])

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
        bulk = self.obs
        bulk_oxidation_state = bulk.composition.oxi_state_guesses()

        if len(bulk_oxidation_state) > 0:
            return bulk_oxidation_state[0]
        else:
            unique_atomic_numbers = np.unique(bulk.atomic_numbers)
            return {chemical_symbols[n]: 0 for n in unique_atomic_numbers}

    def _get_neighborhood_info(self, struc, charge_dict):
        oxi_struc = struc.copy()
        oxi_struc.add_oxidation_state_by_element(charge_dict)
        bulk_equiv = np.array(oxi_struc.site_properties["bulk_equivalent"])
        atomic_numbers = np.array(oxi_struc.atomic_numbers)
        eq_to_Z = np.unique(np.c_[bulk_equiv, atomic_numbers], axis=0)
        eq_to_Z_dict = dict(zip(*eq_to_Z.T))

        Zs = np.unique(oxi_struc.site_properties["bulk_equivalent"])
        combos = combinations_with_replacement(Zs, 2)
        neighbor_dict = {c: None for c in combos}

        neighbor_list = []
        ionic_radii_dict = {Z: [] for Z in Zs}
        coordination_dict = {Z: [] for Z in Zs}

        cnn = CrystalNN(search_cutoff=7.0, cation_anion=True)
        for i, site in enumerate(oxi_struc.sites):
            site_equiv = site.properties["bulk_equivalent"]
            info_dict = cnn.get_nn_info(oxi_struc, i)
            coordination_dict[site.properties["bulk_equivalent"]] = len(
                info_dict
            )
            for neighbor in info_dict:
                neighbor_site_equiv = neighbor["site"].properties[
                    "bulk_equivalent"
                ]
                frac_diff = site.frac_coords - neighbor["site"].frac_coords
                dist = np.linalg.norm(
                    oxi_struc.lattice.get_cartesian_coords(frac_diff)
                )

                species = tuple(sorted([site_equiv, neighbor_site_equiv]))
                neighbor_list.append([species, dist])

        sorted_neighbor_list = sorted(neighbor_list, key=lambda x: x[0])
        groups = groupby(sorted_neighbor_list, key=lambda x: x[0])

        for group in groups:
            nn = list(zip(*group[1]))[1]
            neighbor_dict[group[0]] = np.min(nn)

        for n, d in neighbor_dict.items():
            z1 = eq_to_Z_dict[n[0]]
            z2 = eq_to_Z_dict[n[1]]

            s1 = chemical_symbols[z1]
            s2 = chemical_symbols[z2]
            c1 = charge_dict[s1]
            c2 = charge_dict[s2]

            z1_df = self._ionic_radii_df[
                (self._ionic_radii_df["Atomic Number"] == z1)
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
                d1 = covalent_radii[z1]

            z2_df = self._ionic_radii_df[
                (self._ionic_radii_df["Atomic Number"] == z2)
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
                d2 = covalent_radii[z2]

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

    # def _get_neighborhood_info_old(self, struc, charge_dict):
    #     oxi_struc = struc.copy()
    #     oxi_struc.add_oxidation_state_by_element(charge_dict)
    #     Zs = np.unique(oxi_struc.atomic_numbers)
    #     combos = combinations_with_replacement(Zs, 2)
    #     neighbor_dict = {c: None for c in combos}

    #     neighbor_list = []
    #     ionic_radii_dict = {Z: [] for Z in Zs}
    #     coordination_dict = {Z: [] for Z in Zs}

    #     cnn = CrystalNN(search_cutoff=7.0, cation_anion=True)
    #     for i, site in enumerate(oxi_struc.sites):
    #         info_dict = cnn.get_nn_info(oxi_struc, i)
    #         coordination_dict[site.specie.Z] = len(info_dict)
    #         for neighbor in info_dict:
    #             # dist = site.distance(
    #             #     neighbor["site"],
    #             #     # jimage=neighbor["image"]
    #             # )
    #             frac_diff = site.frac_coords - neighbor["site"].frac_coords
    #             dist = np.linalg.norm(
    #                 oxi_struc.lattice.get_cartesian_coords(frac_diff)
    #             )
    #             species = tuple(
    #                 sorted([site.specie.Z, neighbor["site"].specie.Z])
    #             )
    #             neighbor_list.append([species, dist])

    #     sorted_neighbor_list = sorted(neighbor_list, key=lambda x: x[0])
    #     groups = groupby(sorted_neighbor_list, key=lambda x: x[0])

    #     for group in groups:
    #         nn = list(zip(*group[1]))[1]
    #         print(np.min(nn))
    #         neighbor_dict[group[0]] = np.min(nn)

    #     for n, d in neighbor_dict.items():
    #         s1 = chemical_symbols[n[0]]
    #         s2 = chemical_symbols[n[1]]
    #         c1 = charge_dict[s1]
    #         c2 = charge_dict[s2]

    #         z1_df = self._ionic_radii_df[
    #             (self._ionic_radii_df["Atomic Number"] == n[0])
    #             & (self._ionic_radii_df["Oxidation State"] == c1)
    #         ]

    #         if len(z1_df) > 0:
    #             z1_coords = z1_df["Coordination Number"].values
    #             z1_coord_diff = np.abs(z1_coords - coordination_dict[n[0]])
    #             z1_coord_mask = z1_coord_diff == z1_coord_diff.min()
    #             z1_radii = z1_df[z1_coord_mask]

    #             if not pd.isna(z1_radii["Shannon"]).any():
    #                 d1 = z1_radii["Shannon"].values.mean() / 100
    #             else:
    #                 d1 = z1_radii["ML Mean"].values.mean() / 100
    #         else:
    #             d1 = covalent_radii[n[0]]

    #         z2_df = self._ionic_radii_df[
    #             (self._ionic_radii_df["Atomic Number"] == n[1])
    #             & (self._ionic_radii_df["Oxidation State"] == c2)
    #         ]

    #         if len(z2_df) > 0:
    #             z2_coords = z2_df["Coordination Number"].values
    #             z2_coord_diff = np.abs(z2_coords - coordination_dict[n[1]])
    #             z2_coord_mask = z2_coord_diff == z2_coord_diff.min()
    #             z2_radii = z2_df[z2_coord_mask]

    #             if not pd.isna(z2_radii["Shannon"]).any():
    #                 d2 = z2_radii["Shannon"].values.mean() / 100
    #             else:
    #                 d2 = z2_radii["ML Mean"].values.mean() / 100
    #         else:
    #             d2 = covalent_radii[n[1]]

    #         radius_frac = d1 / (d1 + d2)

    #         if d is None:
    #             neighbor_dict[n] = d1 + d2
    #         else:
    #             r0_1 = radius_frac * d
    #             r0_2 = (1 - radius_frac) * d
    #             ionic_radii_dict[n[0]].append(r0_1)
    #             ionic_radii_dict[n[1]].append(r0_2)

    #     mean_radius_dict = {k: np.mean(v) for k, v in ionic_radii_dict.items()}
    #     print(mean_radius_dict)

    #     return mean_radius_dict

    def _get_r0s(self, bulk, charge_dict):
        bulk_radii_dict = self._get_neighborhood_info(bulk, charge_dict)

        return bulk_radii_dict

    def _calculate(self, inputs: Dict):
        ionic_potential = IonicShiftedForcePotential(
            cutoff=self._cutoff,
        )

        outputs = ionic_potential.forward(
            inputs=inputs,
        )

        return outputs
